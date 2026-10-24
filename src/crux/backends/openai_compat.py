"""OpenAI-compatible chat completions: a vLLM server, OpenRouter or Groq. Retries 429/5xx with backoff."""

from __future__ import annotations

import os
import random
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests

from crux.backends.base import Backend, BackendError, ModelSpec, Registry, load_env


def _is_retryable_status(status: int) -> bool:
    return status == 429 or status >= 500


class OpenAICompatBackend(Backend):
    def __init__(
        self,
        registry: Registry,
        name: str,
        *,
        base_url: str,
        api_key: Optional[str] = None,
        supports_n: bool = True,
        send_chat_template_kwargs: bool = False,
        max_retries: int = 6,
        backoff_base: float = 1.0,
        backoff_max: float = 60.0,
        timeout: float = 120.0,
        session: Any = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        super().__init__(registry)
        self.name = name
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        self.supports_n = supports_n
        self.send_chat_template_kwargs = send_chat_template_kwargs
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self.timeout = timeout
        self.session = session or requests.Session()
        self.sleep = sleep
        self._jitter = random.Random()

    @classmethod
    def from_config(cls, registry: Registry, name: str, **overrides: Any) -> "OpenAICompatBackend":
        """Build from ``backends.<name>`` in models.yaml, reading URLs and keys from the environment / .env."""
        load_env()
        config = registry.backends.get(name) or {}
        if config.get("type") != "openai_compat":
            raise BackendError(f"backend {name!r} is not an openai_compat backend")
        base_url = config.get("base_url") or os.environ.get(config.get("base_url_env", ""), "")
        if not base_url:
            raise BackendError(f"backend {name!r}: set {config.get('base_url_env')} in .env or the environment")
        api_key = os.environ.get(config.get("api_key_env", "")) or None
        return cls(registry, name, base_url=base_url, api_key=api_key,
                   supports_n=bool(config.get("supports_n", True)),
                   send_chat_template_kwargs=bool(config.get("send_chat_template_kwargs", False)),
                   **overrides)

    def _delay(self, attempt: int, response: Any = None) -> float:
        retry_after = response.headers.get("Retry-After") if response is not None else None
        if retry_after:
            try:
                return min(float(retry_after), self.backoff_max)
            except ValueError:
                pass
        return min(self.backoff_max, self.backoff_base * 2 ** attempt) * (0.5 + self._jitter.random() / 2)

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        last_error = ""
        for attempt in range(self.max_retries + 1):
            response = None
            try:
                response = self.session.post(self.url, json=payload, headers=headers, timeout=self.timeout)
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            else:
                if _is_retryable_status(response.status_code):
                    last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                elif response.status_code >= 400:
                    raise BackendError(f"{self.name}: HTTP {response.status_code}: {response.text[:500]}")
                else:
                    body = response.json()
                    error = body.get("error") if isinstance(body, dict) else None
                    if not error:
                        return body
                    code = error.get("code") if isinstance(error, dict) else None
                    if not (isinstance(code, int) and _is_retryable_status(code)):
                        raise BackendError(f"{self.name}: provider error: {error}")
                    last_error = f"provider error: {error}"
            if attempt < self.max_retries:
                self.sleep(self._delay(attempt, response))
        raise BackendError(f"{self.name}: gave up after {self.max_retries + 1} attempts; last error: {last_error}")

    @staticmethod
    def _content(choice: Dict[str, Any]) -> str:
        content = (choice.get("message") or {}).get("content")
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        if not isinstance(content, str) or not content.strip():
            raise BackendError("empty completion")
        return content

    def _generate(self, spec: ModelSpec, messages: List[Dict[str, str]], *, temperature: float,
                  top_p: Optional[float], max_tokens: int, seed: Optional[int],
                  n: int) -> Tuple[List[str], Dict[str, Any]]:
        if spec.chat_template_kwargs and not self.send_chat_template_kwargs:
            # e.g. Qwen3 enable_thinking=False would be silently dropped and thinking would stay on.
            raise BackendError(f"{self.name} cannot apply chat_template_kwargs {spec.chat_template_kwargs} "
                               f"required by {spec.key!r}")
        base = {"model": spec.served_name or spec.hf_repo, "messages": messages,
                "temperature": temperature, "max_tokens": max_tokens}
        if top_p is not None:
            base["top_p"] = top_p
        if spec.chat_template_kwargs:
            base["chat_template_kwargs"] = dict(spec.chat_template_kwargs)

        # Providers that ignore `n` get n separate calls, with seeds seed, seed+1, ...
        if self.supports_n:
            requests_to_send = [dict(base, n=n, **({"seed": seed} if seed is not None else {}))]
        else:
            requests_to_send = [dict(base, **({"seed": seed + i} if seed is not None else {})) for i in range(n)]

        texts: List[str] = []
        served_models, usage = set(), {"prompt_tokens": 0, "completion_tokens": 0}
        for payload in requests_to_send:
            body = self._post(payload)
            choices = body.get("choices") or []
            texts.extend(self._content(choice) for choice in choices)
            served_models.add(body.get("model"))
            for key in usage:
                usage[key] += int((body.get("usage") or {}).get(key) or 0)
        extra = {
            "endpoint_host": urlparse(self.url).netloc,
            "served_model": sorted(m for m in served_models if m),  # what the provider says it ran
            "usage": usage,
        }
        if not self.supports_n and seed is not None:
            extra["seeds"] = [seed + i for i in range(n)]
        return texts, extra
