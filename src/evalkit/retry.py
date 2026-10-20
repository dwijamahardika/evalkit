"""Capped exponential backoff with full jitter.

delay(attempt) ~ Uniform(0, min(cap, base * 2**attempt)), raised to the server's
Retry-After when it sends one. Full jitter (rather than "exponential + small
noise") spreads a burst of synchronised 429s across the whole window, which is
what actually lets a rate-limited endpoint recover.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

from evalkit.providers.base import ProviderError

T = TypeVar("T")

Sleep = Callable[[float], Awaitable[None]]


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 5
    base_s: float = 0.5
    cap_s: float = 30.0
    max_retry_after_s: float = 120.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.base_s < 0 or self.cap_s < 0:
            raise ValueError("backoff durations must be non-negative")


def full_jitter_delay(retry_number: int, policy: RetryPolicy, rng: random.Random) -> float:
    """Delay before retry `retry_number` (0 = first retry)."""
    ceiling = min(policy.cap_s, policy.base_s * (2.0 ** min(retry_number, 60)))
    return rng.uniform(0.0, ceiling)


async def call_with_retries(
    fn: Callable[[], Awaitable[T]],
    policy: RetryPolicy,
    *,
    rng: random.Random | None = None,
    sleep: Sleep = asyncio.sleep,
) -> tuple[T, int]:
    """Run `fn` until it succeeds or a non-retryable error / the attempt budget stops it.

    Returns (result, attempts). On failure the raised ProviderError carries
    `.attempts`.
    """
    rng = rng or random.Random()
    for attempt in range(policy.max_attempts):
        try:
            return await fn(), attempt + 1
        except ProviderError as exc:
            exc.attempts = attempt + 1
            if not exc.retryable or attempt + 1 >= policy.max_attempts:
                raise
            delay = full_jitter_delay(attempt, policy, rng)
            if exc.retry_after_s is not None:
                delay = max(delay, min(exc.retry_after_s, policy.max_retry_after_s))
            await sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover
