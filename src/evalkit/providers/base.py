"""The one interface every model backend implements."""

from __future__ import annotations

import abc
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar, Literal


@dataclass(frozen=True)
class ChatMessage:
    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True)
class CompletionRequest:
    system: str | None
    messages: tuple[ChatMessage, ...]
    sample_index: int
    # Ask the provider for JSON matching this schema (used by the LLM judge).
    response_schema: Mapping[str, Any] | None = None

    def prompt_text(self) -> str:
        """Flat text of the whole prompt, for matching and token estimates."""
        parts = [self.system or ""] + [f"{m.role}: {m.content}" for m in self.messages]
        return "\n\n".join(parts)


@dataclass(frozen=True)
class TokenUsage:
    """Token counts for one billed model invocation.

    `input_tokens` excludes cached prompt tokens, which are counted separately so
    they can be priced separately (providers report them differently; adapters
    normalise to this shape).
    """

    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass(frozen=True)
class Completion:
    text: str
    latency_ms: float
    usage: tuple[TokenUsage, ...]
    served_model: str
    stop_reason: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["usage"] = [asdict(u) for u in self.usage]
        data["extra"] = dict(self.extra)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Completion:
        return cls(
            text=str(data["text"]),
            latency_ms=float(data["latency_ms"]),
            usage=tuple(TokenUsage(**u) for u in data["usage"]),
            served_model=str(data["served_model"]),
            stop_reason=data.get("stop_reason"),
            extra=dict(data.get("extra") or {}),
        )

    def total_usage(self) -> dict[str, int]:
        totals = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        }
        for u in self.usage:
            totals["input_tokens"] += u.input_tokens
            totals["output_tokens"] += u.output_tokens
            totals["cache_read_input_tokens"] += u.cache_read_input_tokens
            totals["cache_creation_input_tokens"] += u.cache_creation_input_tokens
        return totals


def is_retryable_status(status: int) -> bool:
    """408 timeout, 409 conflict, 429 rate limit, and any 5xx (incl. 529 overloaded)."""
    return status in (408, 409, 429) or 500 <= status <= 599


class ProviderError(Exception):
    def __init__(
        self,
        message: str,
        *,
        retryable: bool,
        status: int | None = None,
        retry_after_s: float | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.status = status
        self.retry_after_s = retry_after_s
        self.attempts = 0  # filled in by the retry loop

    def __str__(self) -> str:
        base = super().__str__()
        return f"{base} (HTTP {self.status})" if self.status else base


def parse_retry_after(value: str | None) -> float | None:
    """Seconds from a Retry-After header; HTTP-date values are ignored."""
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        return None
    return seconds if seconds >= 0 else None


class Provider(abc.ABC):
    name: ClassVar[str]

    def __init__(self, model: str) -> None:
        self.model = model

    @abc.abstractmethod
    async def complete(self, request: CompletionRequest) -> Completion:
        """One model call. Raise ProviderError on failure; never retry internally."""

    @abc.abstractmethod
    def identity(self) -> dict[str, Any]:
        """Everything that determines the output distribution. Part of the cache key."""

    async def aclose(self) -> None:  # noqa: B027 - optional hook
        """Release network resources."""
