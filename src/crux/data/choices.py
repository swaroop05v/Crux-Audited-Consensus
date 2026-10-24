"""Parse the base repo's one-string ``choices`` field into an ordered ``{"A": text, ...}`` dict.

Replaces ``run.py``'s ``format_choices_text``, whose A-D-only regex merges options E-J into option D.
"""

from __future__ import annotations

import re
from typing import Any, Dict

# A label "X. " counts only at the start, after a comma, or after a newline, so "vitamin C. " inside text does not.
_LABEL_RE = re.compile(r"(?:^|,\s*|\n\s*)([A-Z])\.\s")


class ChoiceParseError(ValueError):
    pass


def parse_choices(raw: Any) -> Dict[str, str]:
    if isinstance(raw, dict):
        return {str(label): str(text).strip() for label, text in raw.items()}

    text = str(raw or "").strip()
    # Accept only labels that continue the sequence A, B, C, ...; anything else stays part of the option text.
    accepted = []
    for match in _LABEL_RE.finditer(text):
        if match.group(1) == chr(ord("A") + len(accepted)):
            accepted.append(match)
    if len(accepted) < 2:
        raise ChoiceParseError(f"found {len(accepted)} option labels in {text[:80]!r}")

    choices: Dict[str, str] = {}
    for position, match in enumerate(accepted):
        end = accepted[position + 1].start() if position + 1 < len(accepted) else len(text)
        option = text[match.end() : end].strip().rstrip(",").strip()
        if not option:
            raise ChoiceParseError(f"option {match.group(1)} is empty in {text[:80]!r}")
        choices[match.group(1)] = option
    return choices
