"""Model backends. Use ``get_backend`` to build the backend that models.yaml pins for a model."""

from __future__ import annotations

from typing import Any

from crux.backends.base import Backend, BackendError, ChatResult, ModelSpec, Registry, strip_think


def get_backend(registry: Registry, name: str, **overrides: Any) -> Backend:
    """Build backend ``name`` from ``registry.backends``. ``fake`` needs no config entry."""
    if name == "fake":
        from crux.backends.fake import FakeBackend

        return FakeBackend(registry, **overrides)
    config = registry.backends.get(name)
    if config is None:
        raise BackendError(f"unknown backend {name!r}; known: {sorted(registry.backends)} and 'fake'")
    if config.get("type") == "openai_compat":
        from crux.backends.openai_compat import OpenAICompatBackend

        return OpenAICompatBackend.from_config(registry, name, **overrides)
    if config.get("type") == "hf_local":
        from crux.backends.hf_local import HFLocalBackend

        return HFLocalBackend(registry, **overrides)
    raise BackendError(f"backend {name!r} has unknown type {config.get('type')!r}")


__all__ = ["Backend", "BackendError", "ChatResult", "ModelSpec", "Registry", "get_backend", "strip_think"]
