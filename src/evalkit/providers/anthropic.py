"""Anthropic Messages API through the official `anthropic` SDK."""

from __future__ import annotations

import time
from typing import Any

import anthropic
from anthropic.types.beta import BetaMessage, BetaOutputConfigParam

from evalkit.config import AnthropicProviderConfig
from evalkit.providers.base import (
    Completion,
    CompletionRequest,
    Provider,
    ProviderError,
    TokenUsage,
    is_retryable_status,
    parse_retry_after,
)

# Server-side refusal fallback ("fallbacks": "default") re-runs a declined request on
# a model Anthropic picks for the refusal category. Only some models accept it.
FALLBACK_BETA = "server-side-fallback-2026-07-01"
FALLBACK_CAPABLE_MODELS = frozenset(
    {"claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-fable-5", "claude-sonnet-5-5"}
)


def fallbacks_enabled(cfg: AnthropicProviderConfig) -> bool:
    if cfg.fallbacks is None:
        return cfg.model in FALLBACK_CAPABLE_MODELS
    return cfg.fallbacks == "default"


def _usage(message: BetaMessage, requested_model: str) -> tuple[TokenUsage, ...]:
    """Per-model token usage.

    With a server-side fallback, `usage.iterations` is the per-attempt source of
    truth (each attempt is billed at its own model's rates); top-level `usage`
    only covers the attempt that produced the message.
    """
    iterations = message.usage.iterations or []
    if iterations:
        return tuple(
            TokenUsage(
                model=str(getattr(it, "model", None) or requested_model),
                input_tokens=it.input_tokens,
                output_tokens=it.output_tokens,
                cache_read_input_tokens=it.cache_read_input_tokens or 0,
                cache_creation_input_tokens=it.cache_creation_input_tokens or 0,
            )
            for it in iterations
            if it.type in ("message", "fallback_message")
        )
    u = message.usage
    return (
        TokenUsage(
            model=str(message.model),
            input_tokens=u.input_tokens,
            output_tokens=u.output_tokens,
            cache_read_input_tokens=u.cache_read_input_tokens or 0,
            cache_creation_input_tokens=u.cache_creation_input_tokens or 0,
        ),
    )


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(
        self, cfg: AnthropicProviderConfig, client: anthropic.AsyncAnthropic | None = None
    ) -> None:
        super().__init__(cfg.model)
        self.cfg = cfg
        # max_retries=0: evalkit's own retry loop owns backoff, so a 429 is never
        # retried 3 (SDK) x 5 (evalkit) times, and attempts are counted in the run.
        self._client = client or anthropic.AsyncAnthropic(max_retries=0, timeout=cfg.timeout_s)
        self._fallbacks = fallbacks_enabled(cfg)

    def identity(self) -> dict[str, Any]:
        return {
            "type": self.name,
            "base_url": str(self._client.base_url),
            "model": self.cfg.model,
            "max_tokens": self.cfg.max_tokens,
            "effort": self.cfg.effort,
            "fallbacks": self._fallbacks,
        }

    async def complete(self, request: CompletionRequest) -> Completion:
        output_config: BetaOutputConfigParam = {}
        if self.cfg.effort is not None:
            output_config["effort"] = self.cfg.effort
        if request.response_schema is not None:
            output_config["format"] = {
                "type": "json_schema",
                "schema": dict(request.response_schema),
            }
        started = time.perf_counter()
        try:
            message = await self._client.beta.messages.create(
                model=self.cfg.model,
                max_tokens=self.cfg.max_tokens,
                system=request.system if request.system else anthropic.omit,
                messages=[{"role": m.role, "content": m.content} for m in request.messages],
                output_config=output_config or anthropic.omit,
                fallbacks="default" if self._fallbacks else anthropic.omit,
                betas=[FALLBACK_BETA] if self._fallbacks else anthropic.omit,
            )
        except anthropic.RateLimitError as exc:
            raise ProviderError(
                f"anthropic: rate limited: {exc.message}",
                retryable=True,
                status=exc.status_code,
                retry_after_s=parse_retry_after(exc.response.headers.get("retry-after")),
            ) from exc
        except anthropic.APIStatusError as exc:
            raise ProviderError(
                f"anthropic: {exc.message}",
                retryable=is_retryable_status(exc.status_code),
                status=exc.status_code,
                retry_after_s=parse_retry_after(exc.response.headers.get("retry-after")),
            ) from exc
        except anthropic.APITimeoutError as exc:
            raise ProviderError("anthropic: request timed out", retryable=True) from exc
        except anthropic.APIConnectionError as exc:
            raise ProviderError(f"anthropic: connection error: {exc}", retryable=True) from exc
        latency_ms = (time.perf_counter() - started) * 1000.0

        # Thinking blocks (empty by default) and fallback markers are not output.
        text = "".join(block.text for block in message.content if block.type == "text")
        extra: dict[str, Any] = {}
        if message.stop_reason == "refusal" and message.stop_details is not None:
            extra["refusal_category"] = message.stop_details.category
        return Completion(
            text=text,
            latency_ms=latency_ms,
            usage=_usage(message, self.cfg.model),
            served_model=str(message.model),
            stop_reason=message.stop_reason,
            extra=extra,
        )

    async def aclose(self) -> None:
        await self._client.close()
