from __future__ import annotations

import asyncio
import random
from collections.abc import Coroutine
from pathlib import Path
from typing import Any, TypeVar

import pytest
from hypothesis import given
from hypothesis import strategies as st

from evalkit import template
from evalkit.cache import CacheMode, ResponseCache, cache_key
from evalkit.providers.base import (
    ChatMessage,
    Completion,
    CompletionRequest,
    ProviderError,
    TokenUsage,
    parse_retry_after,
)
from evalkit.retry import RetryPolicy, call_with_retries, full_jitter_delay

T = TypeVar("T")

# ------------------------------------------------------------------- template


def test_render_basic_and_json_values() -> None:
    out = template.render(
        "Hi {{ name }}, {{n}} items: {{ items }}", {"name": "Ana", "n": 3, "items": [1, 2]}
    )
    assert out == "Hi Ana, 3 items: [1, 2]"
    assert template.variables("{{a}} {{ b }} {{a}}") == {"a", "b"}


def test_render_missing_variable() -> None:
    with pytest.raises(template.TemplateError):
        template.render("{{ missing }}", {})


@given(st.text())
def test_inputs_are_never_re_expanded(value: str) -> None:
    # Single-pass substitution: a value that itself looks like a placeholder is
    # inserted verbatim (no template injection from test data).
    assert template.render("{{ a }}", {"a": value, "b": "SHOULD-NOT-APPEAR"}) == value
    assert template.render("<{{a}}>", {"a": "{{ b }}", "b": "x"}) == "<{{ b }}>"


# ---------------------------------------------------------------------- retry


@given(
    st.integers(min_value=0, max_value=200),
    st.floats(min_value=0.0, max_value=10.0),
    st.floats(min_value=0.0, max_value=100.0),
    st.integers(),
)
def test_full_jitter_bounds(retry: int, base: float, cap: float, seed: int) -> None:
    policy = RetryPolicy(base_s=base, cap_s=cap)
    delay = full_jitter_delay(retry, policy, random.Random(seed))
    assert 0.0 <= delay <= min(cap, base * 2.0 ** min(retry, 60)) + 1e-12


class Flaky:
    def __init__(self, failures: list[ProviderError]) -> None:
        self.failures = failures
        self.calls = 0

    async def __call__(self) -> str:
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return "ok"


def _run(coro: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coro)


def test_retries_until_success_and_counts_attempts() -> None:
    sleeps: list[float] = []

    async def sleep(d: float) -> None:
        sleeps.append(d)

    fn = Flaky(
        [
            ProviderError("x", retryable=True, status=429),
            ProviderError("y", retryable=True, status=503),
        ]
    )
    result, attempts = _run(
        call_with_retries(fn, RetryPolicy(base_s=1.0, cap_s=8.0), rng=random.Random(1), sleep=sleep)
    )
    assert (result, attempts, fn.calls) == ("ok", 3, 3)
    assert len(sleeps) == 2
    assert 0.0 <= sleeps[0] <= 1.0
    assert 0.0 <= sleeps[1] <= 2.0


def test_non_retryable_fails_fast() -> None:
    fn = Flaky([ProviderError("bad request", retryable=False, status=400)])
    with pytest.raises(ProviderError) as info:
        _run(call_with_retries(fn, RetryPolicy(), sleep=_no_sleep))
    assert info.value.attempts == 1
    assert fn.calls == 1


def test_gives_up_after_max_attempts() -> None:
    fn = Flaky([ProviderError("down", retryable=True, status=503) for _ in range(10)])
    with pytest.raises(ProviderError) as info:
        _run(call_with_retries(fn, RetryPolicy(max_attempts=4), sleep=_no_sleep))
    assert info.value.attempts == 4
    assert fn.calls == 4


def test_retry_after_is_honoured_and_capped() -> None:
    sleeps: list[float] = []

    async def sleep(d: float) -> None:
        sleeps.append(d)

    fn = Flaky(
        [
            ProviderError("slow down", retryable=True, status=429, retry_after_s=7.0),
            ProviderError("slow down", retryable=True, status=429, retry_after_s=9999.0),
        ]
    )
    policy = RetryPolicy(base_s=0.1, cap_s=0.2, max_retry_after_s=60.0)
    _run(call_with_retries(fn, policy, sleep=sleep))
    assert sleeps == [7.0, 60.0]


def test_parse_retry_after() -> None:
    assert parse_retry_after("3") == 3.0
    assert parse_retry_after("1.5") == 1.5
    assert parse_retry_after("Wed, 21 Oct 2026 07:28:00 GMT") is None
    assert parse_retry_after("-1") is None
    assert parse_retry_after(None) is None


async def _no_sleep(_: float) -> None:
    return None


# ---------------------------------------------------------------------- cache


def _request(sample: int = 0, user: str = "hi", system: str | None = "sys") -> CompletionRequest:
    return CompletionRequest(
        system=system, messages=(ChatMessage("user", user),), sample_index=sample
    )


def _completion(text: str = "out") -> Completion:
    return Completion(
        text=text,
        latency_ms=12.5,
        usage=(TokenUsage(model="m", input_tokens=3, output_tokens=4, cache_read_input_tokens=1),),
        served_model="m",
        stop_reason="end_turn",
        extra={"k": "v"},
    )


def test_cache_key_is_order_independent_and_sensitive() -> None:
    a = {"type": "fake", "model": "m", "params": {"x": 1, "y": 2}}
    b = {"params": {"y": 2, "x": 1}, "model": "m", "type": "fake"}
    assert cache_key(a, _request()) == cache_key(b, _request())
    base = cache_key(a, _request())
    assert cache_key(a, _request(sample=1)) != base
    assert cache_key(a, _request(user="other")) != base
    assert cache_key(a, _request(system=None)) != base
    assert cache_key({**a, "model": "m2"}, _request()) != base


def test_cache_roundtrip_and_modes(tmp_path: Path) -> None:
    key = cache_key({"model": "m"}, _request())
    cache = ResponseCache(tmp_path, CacheMode.USE)
    assert cache.get(key) is None
    cache.put(key, _completion())
    assert cache.get(key) == _completion()
    assert (cache.hits, cache.misses) == (1, 1)

    refresh = ResponseCache(tmp_path, CacheMode.REFRESH)
    assert refresh.get(key) is None  # ignores existing entries...
    refresh.put(key, _completion("new"))  # ...but writes fresh ones
    assert ResponseCache(tmp_path).get(key) == _completion("new")

    off = ResponseCache(tmp_path / "off", CacheMode.OFF)
    off.put(key, _completion())
    assert not (tmp_path / "off").exists()


def test_corrupt_cache_entry_is_a_miss(tmp_path: Path) -> None:
    key = cache_key({"model": "m"}, _request())
    cache = ResponseCache(tmp_path)
    cache.put(key, _completion())
    path = next(tmp_path.rglob(f"{key}.json"))
    path.write_text("{ truncated", encoding="utf-8")
    assert cache.get(key) is None
    assert not list(tmp_path.rglob(".tmp-*"))  # atomic writes leave no temp files
