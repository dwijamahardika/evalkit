"""Generic provider for servers that implement OpenAI-style /chat/completions.

Covers hosted APIs and local inference servers alike; only the wire format is
assumed. The API key, if any, is read from the environment variable named in
the suite (`api_key_env`), never from the suite itself.
"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx2

from evalkit.config import OpenAIProviderConfig
from evalkit.providers.base import (
    Completion,
    CompletionRequest,
    Provider,
    ProviderError,
    TokenUsage,
    is_retryable_status,
    parse_retry_after,
)


class MissingCredentialError(RuntimeError):
    pass


class OpenAICompatProvider(Provider):
    name = "openai"

    def __init__(self, cfg: OpenAIProviderConfig, client: httpx2.AsyncClient | None = None) -> None:
        super().__init__(cfg.model)
        self.cfg = cfg
        headers = {"content-type": "application/json"}
        if cfg.api_key_env:
            key = os.environ.get(cfg.api_key_env)
            if not key:
                raise MissingCredentialError(
                    f"environment variable {cfg.api_key_env} is not set "
                    "(set api_key_env: null for servers without auth)"
                )
            headers["authorization"] = f"Bearer {key}"
        self._headers = headers
        self._client = client or httpx2.AsyncClient(timeout=cfg.timeout_s)

    def identity(self) -> dict[str, Any]:
        return {
            "type": self.name,
            "base_url": self.cfg.base_url,
            "model": self.cfg.model,
            "max_tokens": self.cfg.max_tokens,
            "temperature": self.cfg.temperature,
            "top_p": self.cfg.top_p,
            "seed": self.cfg.seed,
        }

    def _body(self, request: CompletionRequest) -> dict[str, Any]:
        messages: list[dict[str, str]] = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages += [{"role": m.role, "content": m.content} for m in request.messages]
        body: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            self.cfg.max_tokens_field: self.cfg.max_tokens,
        }
        for key in ("temperature", "top_p", "seed"):
            value = getattr(self.cfg, key)
            if value is not None:
                body[key] = value
        if request.response_schema is not None and self.cfg.structured_output:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": dict(request.response_schema)},
            }
        return body

    async def complete(self, request: CompletionRequest) -> Completion:
        url = f"{self.cfg.base_url}/chat/completions"
        started = time.perf_counter()
        try:
            resp = await self._client.post(url, json=self._body(request), headers=self._headers)
        except httpx2.TimeoutException as exc:
            raise ProviderError("openai: request timed out", retryable=True) from exc
        except httpx2.TransportError as exc:
            raise ProviderError(f"openai: transport error: {exc}", retryable=True) from exc
        latency_ms = (time.perf_counter() - started) * 1000.0
        if resp.status_code >= 400:
            raise ProviderError(
                f"openai: {resp.text[:200]}",
                retryable=is_retryable_status(resp.status_code),
                status=resp.status_code,
                retry_after_s=parse_retry_after(resp.headers.get("retry-after")),
            )
        try:
            data = resp.json()
            choice = data["choices"][0]
            text = choice["message"].get("content") or ""
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"openai: malformed response: {exc!r}", retryable=False) from exc
        usage = data.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
        return Completion(
            text=str(text),
            latency_ms=latency_ms,
            # Priced under the configured name: servers often answer with a dated alias.
            usage=(
                TokenUsage(
                    model=self.cfg.model,
                    input_tokens=max(0, prompt_tokens - cached),
                    output_tokens=int(usage.get("completion_tokens") or 0),
                    cache_read_input_tokens=cached,
                ),
            ),
            served_model=str(data.get("model") or self.cfg.model),
            stop_reason=choice.get("finish_reason"),
        )

    async def aclose(self) -> None:
        await self._client.aclose()
