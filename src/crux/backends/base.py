"""Backend interface, model registry, and resolved-parameter logging.

Every model call goes through ``Backend.chat``. It returns a ``ChatResult`` whose ``params`` hold the fully
resolved call (model repo and pinned revision, backend, dtype, quantization, sampling, seed, n, template kwargs,
plus backend extras). Stages put these into their ``meta``:

    result = backend.chat("consultant_a", messages, **registry.sampling("consultant"), max_tokens=1024, seed=7)
    meta = stage_meta("cruxes", calls={"proposal": result.params}, **prompt.meta())
"""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from crux.io import REPO_ROOT

DEFAULT_REGISTRY_PATH = REPO_ROOT / "configs" / "models.yaml"
MESSAGE_ROLES = {"system", "user", "assistant"}
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")


class BackendError(RuntimeError):
    pass


def strip_think(text: str) -> str:
    """Drop ``<think>...</think>`` blocks; braces inside them would break JSON extraction."""
    return _THINK_RE.sub("", text).strip()


def load_env(path: Path = REPO_ROOT / ".env") -> None:
    """Read KEY=VALUE lines from the gitignored .env into os.environ, never overriding variables already set."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_repo: str
    revision: str
    backend: str
    dtype: str
    quantization: str
    served_name: Optional[str] = None
    chat_template_kwargs: Dict[str, Any] = field(default_factory=dict)
    strip_think: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    def describe(self) -> Dict[str, Any]:
        return {
            "model_key": self.key,
            "hf_repo": self.hf_repo,
            "revision": self.revision,
            "backend": self.backend,
            "dtype": self.dtype,
            "quantization": self.quantization,
            "chat_template_kwargs": dict(self.chat_template_kwargs),
            **({"served_name": self.served_name} if self.served_name else {}),
            **self.extra,
        }


_SPEC_FIELDS = {"hf_repo", "revision", "backend", "dtype", "quantization", "served_name",
                "chat_template_kwargs", "strip_think"}


class Registry:
    """Models, backends and sampling profiles from configs/models.yaml."""

    def __init__(self, models: Dict[str, ModelSpec], backends: Dict[str, Dict[str, Any]],
                 sampling: Dict[str, Dict[str, Any]]):
        self.models = models
        self.backends = backends
        self._sampling = sampling

    @classmethod
    def load(cls, path: Path = DEFAULT_REGISTRY_PATH) -> "Registry":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        backends = raw.get("backends") or {}
        models = {}
        for key, entry in (raw.get("models") or {}).items():
            for required in ("hf_repo", "revision", "backend", "dtype", "quantization"):
                if not entry.get(required):
                    raise BackendError(f"models.{key}.{required} is required")
            if not _REVISION_RE.match(str(entry["revision"])):
                raise BackendError(f"models.{key}.revision must be a pinned 40-char commit hash")
            if entry["backend"] not in backends:
                raise BackendError(f"models.{key}.backend {entry['backend']!r} is not defined under backends")
            models[key] = ModelSpec(
                key=key,
                hf_repo=entry["hf_repo"],
                revision=str(entry["revision"]),
                backend=entry["backend"],
                dtype=entry["dtype"],
                quantization=entry["quantization"],
                served_name=entry.get("served_name"),
                chat_template_kwargs=dict(entry.get("chat_template_kwargs") or {}),
                strip_think=bool(entry.get("strip_think", False)),
                extra={k: v for k, v in entry.items() if k not in _SPEC_FIELDS},
            )
        return cls(models, backends, raw.get("sampling") or {})

    def without_think_stripping(self) -> "Registry":
        """A copy whose models keep ``<think>`` blocks, for checking that thinking is really off."""
        models = {key: replace(spec, strip_think=False) for key, spec in self.models.items()}
        return Registry(models, self.backends, self._sampling)

    def spec(self, model_key: str) -> ModelSpec:
        if model_key not in self.models:
            raise BackendError(f"unknown model key {model_key!r}; known: {sorted(self.models)}")
        return self.models[model_key]

    def sampling(self, profile: str) -> Dict[str, Any]:
        """Sampling for a call type (decisions D3), as keyword arguments for ``Backend.chat``."""
        if profile not in self._sampling:
            raise BackendError(f"unknown sampling profile {profile!r}; known: {sorted(self._sampling)}")
        values = self._sampling[profile]
        return {"temperature": float(values["temperature"]), "top_p": values.get("top_p")}


@dataclass
class ChatResult:
    texts: List[str]
    params: Dict[str, Any]

    @property
    def text(self) -> str:
        if len(self.texts) != 1:
            raise BackendError(f"ChatResult has {len(self.texts)} completions; use .texts")
        return self.texts[0]

    @property
    def finish_reason(self) -> Optional[str]:
        """Finish reason of the single completion; "length" means it was cut off at max_tokens."""
        return self.params["finish_reasons"][0] if len(self.texts) == 1 else None


def validate_messages(messages: Any) -> None:
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    for position, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") not in MESSAGE_ROLES:
            raise ValueError(f"messages[{position}]: role must be one of {sorted(MESSAGE_ROLES)}")
        if not isinstance(message.get("content"), str):
            raise ValueError(f"messages[{position}]: content must be a string")


class Backend(ABC):
    """Base class. Subclasses implement ``_generate``; ``chat`` validates, post-processes and logs."""

    name: str = "base"

    def __init__(self, registry: Registry):
        self.registry = registry

    def spec(self, model_key: str) -> ModelSpec:
        spec = self.registry.spec(model_key)
        if spec.backend != self.name:
            raise BackendError(
                f"model {model_key!r} is pinned to backend {spec.backend!r}, not {self.name!r} "
                "(one backend per model for the whole kill test)"
            )
        return spec

    def chat(
        self,
        model_key: str,
        messages: List[Dict[str, str]],
        *,
        temperature: float,
        top_p: Optional[float],
        max_tokens: int,
        seed: Optional[int],
        n: int = 1,
    ) -> ChatResult:
        validate_messages(messages)
        if temperature < 0:
            raise ValueError("temperature must be >= 0")
        if top_p is not None and not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1] or None")
        if max_tokens < 1 or n < 1:
            raise ValueError("max_tokens and n must be >= 1")
        if seed is not None and not isinstance(seed, int):
            raise ValueError("seed must be an int or None")

        spec = self.spec(model_key)
        texts, extra = self._generate(spec, messages, temperature=temperature, top_p=top_p,
                                      max_tokens=max_tokens, seed=seed, n=n)
        if len(texts) != n:
            raise BackendError(f"{self.name}: expected {n} completions, got {len(texts)}")
        if spec.strip_think:
            texts = [strip_think(text) for text in texts]
        # One finish reason per completion ("stop", "length", ... or None when the backend cannot tell).
        finish_reasons = list(extra.pop("finish_reasons", None) or [None] * n)
        params = {
            **spec.describe(),
            "backend": self.name,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "seed": seed,
            "n": n,
            **extra,
            "finish_reasons": finish_reasons,
            "truncated": any(reason == "length" for reason in finish_reasons),
        }
        return ChatResult(texts=texts, params=params)

    @abstractmethod
    def _generate(self, spec: ModelSpec, messages: List[Dict[str, str]], *, temperature: float,
                  top_p: Optional[float], max_tokens: int, seed: Optional[int],
                  n: int) -> Tuple[List[str], Dict[str, Any]]:
        """Return (completions, backend-specific params to log)."""

    def verdict_probs(self, model_key: str, messages: List[Dict[str, str]], completion: str,
                      *, marker: str = "VERDICT:") -> Dict[str, Any]:
        """Next-token TRUE/FALSE probabilities after ``marker`` (audit 4.2). Needs the full distribution."""
        raise NotImplementedError(f"{self.name} cannot return full next-token distributions")
