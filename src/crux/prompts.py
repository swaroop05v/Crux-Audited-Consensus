"""Load versioned prompt files from prompts/<version>/<name>.txt and render them into chat messages.

Prompt files follow Doc 2: an optional ``SYSTEM:`` section and a ``USER:`` section, each header on its own line.
Placeholders are ``{lowercase_name}``; JSON braces in the templates are left alone. Rendering is strict: every
placeholder must be filled and no unknown field may be passed.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from crux.io import REPO_ROOT

PROMPTS_DIR = REPO_ROOT / "prompts"
PLACEHOLDER_RE = re.compile(r"\{([a-z][a-z0-9_]*)\}")
SECTION_HEADERS = {"SYSTEM:": "system", "USER:": "user"}


class PromptError(ValueError):
    pass


@dataclass(frozen=True)
class Prompt:
    name: str
    version: str
    text: str
    sha256: str
    path: Path

    @property
    def placeholders(self) -> List[str]:
        return sorted(set(PLACEHOLDER_RE.findall(self.text)))

    def render(self, **fields: str) -> str:
        missing = set(self.placeholders) - set(fields)
        unknown = set(fields) - set(self.placeholders)
        if missing or unknown:
            raise PromptError(f"{self.name}: missing fields {sorted(missing)}, unknown fields {sorted(unknown)}")
        # One pass, so braces inside field values (e.g. JSON in a reasoning trace) are never re-substituted.
        return PLACEHOLDER_RE.sub(lambda match: str(fields[match.group(1)]), self.text)

    def messages(self, **fields: str) -> List[Dict[str, str]]:
        """Render and split into chat messages by section header. A file without headers is one user message."""
        rendered = self.render(**fields)
        messages: List[Dict[str, str]] = []
        role, lines = None, []
        for line in rendered.split("\n"):
            if line.strip() in SECTION_HEADERS:
                if role is not None:
                    messages.append({"role": role, "content": "\n".join(lines).strip()})
                role, lines = SECTION_HEADERS[line.strip()], []
            else:
                lines.append(line)
        if role is None:
            return [{"role": "user", "content": rendered.strip()}]
        messages.append({"role": role, "content": "\n".join(lines).strip()})
        return messages

    def meta(self) -> Dict[str, str]:
        """Fields for a stage ``meta`` block."""
        return {
            "prompt_name": self.name,
            "prompt_version": self.version,
            "prompt_file": self.path.relative_to(REPO_ROOT).as_posix(),
            "prompt_sha256": self.sha256,
        }


def load_prompt(name: str, version: str = "v1") -> Prompt:
    path = PROMPTS_DIR / version / f"{name}.txt"
    if not path.exists():
        raise PromptError(f"prompt file not found: {path}")
    # Normalize line endings so the hash is the same on every OS (see .gitattributes).
    text = path.read_bytes().decode("utf-8").replace("\r\n", "\n")
    sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return Prompt(name=name, version=version, text=text, sha256=sha256, path=path)
