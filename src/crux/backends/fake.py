"""Deterministic backend for tests and offline plumbing. Never touches a network or a GPU."""

from __future__ import annotations

import copy
import json
import threading
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from crux.backends.base import Backend, ModelSpec, Registry

DEFAULT_RESPONSE = json.dumps({"answer": "A", "reasoning": ["fake step"]})

Responder = Union[str, Sequence[str], Callable[[str, List[Dict[str, str]], int], str]]


class FakeBackend(Backend):
    """Returns canned text.

    ``responder`` is either a fixed string, a sequence cycled call by call, or a function
    ``(model_key, messages, call_index) -> str``. Accepts every model in the registry regardless of its pinned
    backend, so tests can use the real model keys. Every call is recorded in ``self.calls``.
    """

    name = "fake"

    def __init__(self, registry: Registry, responder: Responder = DEFAULT_RESPONSE,
                 verdict: Optional[Dict[str, float]] = None,
                 finish_reason: Union[str, Callable[[str, int], str]] = "stop"):
        """``finish_reason`` is fixed, or ``(model_key, call_index) -> str`` (use "length" to simulate truncation)."""
        super().__init__(registry)
        self.responder = responder
        self.verdict = verdict or {"p_true_raw": 0.6, "p_false_raw": 0.2}
        self.finish_reason = finish_reason
        self.calls: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def spec(self, model_key: str) -> ModelSpec:
        return self.registry.spec(model_key)

    def _respond(self, model_key: str, messages: List[Dict[str, str]]) -> str:
        index = len(self.calls)
        if callable(self.responder):
            return self.responder(model_key, messages, index)
        if isinstance(self.responder, str):
            return self.responder
        return self.responder[index % len(self.responder)]

    def _generate(self, spec: ModelSpec, messages: List[Dict[str, str]], *, temperature: float,
                  top_p: Optional[float], max_tokens: int, seed: Optional[int],
                  n: int) -> Tuple[List[str], Dict[str, Any]]:
        texts, finish_reasons = [], []
        with self._lock:  # stages may call from several threads (--workers)
            for _ in range(n):
                index = len(self.calls)
                texts.append(self._respond(spec.key, messages))
                finish_reasons.append(self.finish_reason(spec.key, index) if callable(self.finish_reason)
                                      else self.finish_reason)
                self.calls.append({"model_key": spec.key, "messages": copy.deepcopy(messages),
                                   "temperature": temperature, "top_p": top_p, "max_tokens": max_tokens,
                                   "seed": seed})
        return texts, {"finish_reasons": finish_reasons}

    def verdict_probs(self, model_key: str, messages: List[Dict[str, str]], completion: str,
                      *, marker: str = "VERDICT:") -> Dict[str, Any]:
        from crux.backends.hf_local import cut_after_marker, verdict_readout

        cut_after_marker(completion, marker)  # same failure mode as the real backend when the marker is missing
        self.registry.spec(model_key)
        probs = {0: self.verdict["p_true_raw"], 1: self.verdict["p_false_raw"]}
        return verdict_readout(lambda token_id: probs.get(token_id, 0.0), [0], [1])
