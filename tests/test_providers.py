"""Provider adapters, tested offline against mocked HTTP transports."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar

import anthropic
import httpx2
import pytest

from evalkit.config import AnthropicProviderConfig, OpenAIProviderConfig
from evalkit.judge import JUDGE_SCHEMA
from evalkit.providers.anthropic import FALLBACK_BETA, AnthropicProvider
from evalkit.providers.base import ChatMessage, CompletionRequest, ProviderError
from evalkit.providers.fake import FakeProvider, FakeScript
from evalkit.providers.openai_compat import MissingCredentialError, OpenAICompatProvider

T = TypeVar("T")


def run(coro: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coro)


def req(
    user: str = "hello", sample: int = 0, schema: dict[str, Any] | None = None
) -> CompletionRequest:
    return CompletionRequest(
        system="be terse",
        messages=(ChatMessage("user", user),),
        sample_index=sample,
        response_schema=schema,
    )


# ----------------------------------------------------------------------- fake


def fake(script: dict[str, Any], model: str = "m") -> FakeProvider:
    return FakeProvider(model, FakeScript.model_validate(script))


def test_fake_cycles_by_sample_index_with_repeat() -> None:
    p = fake({"rules": [{"when": "hello", "responses": [{"text": "A", "repeat": 2}, "B"]}]})
    texts = [run(p.complete(req(sample=i))).text for i in range(6)]
    assert texts == ["A", "A", "B", "A", "A", "B"]


def test_fake_rule_order_model_filter_regex_and_default() -> None:
    p_a = fake(
        {
            "rules": [
                {"model": "b", "when": "hello", "responses": ["only-b"]},
                {"when_regex": r"hel+o", "responses": ["regex"]},
            ],
            "default": "dflt",
        },
        model="a",
    )
    assert run(p_a.complete(req("hello"))).text == "regex"
    assert run(p_a.complete(req("bye"))).text == "dflt"
    p_b = fake({"rules": [{"model": "b", "when": "hello", "responses": ["only-b"]}]}, model="b")
    assert run(p_b.complete(req("hello"))).text == "only-b"
    with pytest.raises(ProviderError) as info:
        run(p_b.complete(req("bye")))
    assert info.value.retryable is False


def test_fake_scripted_failures_then_success() -> None:
    p = fake({"rules": [{"when": "hello", "fail_attempts": [429, 400], "responses": ["ok"]}]})
    with pytest.raises(ProviderError) as first:
        run(p.complete(req()))
    assert (first.value.status, first.value.retryable) == (429, True)
    with pytest.raises(ProviderError) as second:
        run(p.complete(req()))
    assert (second.value.status, second.value.retryable) == (400, False)
    assert run(p.complete(req())).text == "ok"
    # Failures are per (prompt, sample): another sample starts from attempt 0.
    with pytest.raises(ProviderError):
        run(p.complete(req(sample=1)))


def test_fake_identity_tracks_script_content() -> None:
    a = fake({"rules": [{"responses": ["x"]}]})
    b = fake({"rules": [{"responses": ["y"]}]})
    assert a.identity() != b.identity()
    assert a.identity() == fake({"rules": [{"responses": ["x"]}]}).identity()


# ------------------------------------------------------------------ anthropic

Handler = Callable[[httpx2.Request], httpx2.Response]


def anthropic_provider(handler: Handler, **cfg: Any) -> AnthropicProvider:
    client = anthropic.AsyncAnthropic(
        api_key="offline-test",
        base_url="http://anthropic.test",
        max_retries=0,
        http_client=anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handler)),
    )
    return AnthropicProvider(AnthropicProviderConfig(type="anthropic", **cfg), client=client)


def message(
    text: str = '{"ok": true}',
    *,
    model: str = "claude-opus-5-5",
    stop_reason: str = "end_turn",
    usage: dict[str, Any] | None = None,
    content: list[dict[str, Any]] | None = None,
    stop_details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    blocks = (
        content
        if content is not None
        else [
            {"type": "thinking", "thinking": "", "signature": "sig"},
            {"type": "text", "text": text},
        ]
    )
    return {
        "id": "msg_offline",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "stop_details": stop_details,
        "usage": usage
        or {
            "input_tokens": 11,
            "output_tokens": 7,
            "cache_read_input_tokens": 3,
            "cache_creation_input_tokens": 2,
        },
    }


class Recorder:
    def __init__(self, *responses: httpx2.Response) -> None:
        self.responses = list(responses)
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        return self.responses.pop(0)

    def body(self, i: int = 0) -> dict[str, Any]:
        data: dict[str, Any] = json.loads(self.requests[i].content)
        return data


def test_anthropic_request_shape_and_parsing() -> None:
    rec = Recorder(httpx2.Response(200, json=message("hello world")))
    p = anthropic_provider(rec, effort="low")
    out = run(p.complete(req(schema=JUDGE_SCHEMA)))
    body = rec.body()
    assert rec.requests[0].url.path == "/v1/messages"
    assert body["model"] == "claude-opus-5-5"
    assert body["max_tokens"] == 16_000
    assert body["system"] == "be terse"
    assert body["messages"] == [{"role": "user", "content": "hello"}]
    assert body["output_config"] == {
        "effort": "low",
        "format": {"type": "json_schema", "schema": JUDGE_SCHEMA},
    }
    assert "temperature" not in body  # rejected by current Claude models
    assert body["fallbacks"] == "default"
    assert FALLBACK_BETA in rec.requests[0].headers["anthropic-beta"]
    # thinking blocks are not output; usage is normalised
    assert out.text == "hello world"
    assert out.served_model == "claude-opus-5-5"
    assert out.stop_reason == "end_turn"
    (u,) = out.usage
    assert (
        u.input_tokens,
        u.output_tokens,
        u.cache_read_input_tokens,
        u.cache_creation_input_tokens,
    ) == (11, 7, 3, 2)


def test_anthropic_fallbacks_off_for_other_models_and_when_disabled() -> None:
    for cfg in ({"model": "claude-haiku-5-5"}, {"fallbacks": "none"}):
        rec = Recorder(httpx2.Response(200, json=message()))
        run(anthropic_provider(rec, **cfg).complete(req()))
        body = rec.body()
        assert "fallbacks" not in body
        assert "output_config" not in body
        assert FALLBACK_BETA not in rec.requests[0].headers.get("anthropic-beta", "")


def test_anthropic_fallback_usage_is_per_model() -> None:
    usage = {
        "input_tokens": 100,
        "output_tokens": 40,
        "iterations": [
            {
                "type": "message",
                "model": "claude-opus-5-5",
                "input_tokens": 100,
                "output_tokens": 0,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
            {
                "type": "fallback_message",
                "model": "claude-opus-4-8",
                "input_tokens": 100,
                "output_tokens": 40,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
        ],
    }
    content: list[dict[str, Any]] = [
        {
            "type": "fallback",
            "from": {"model": "claude-opus-5-5"},
            "to": {"model": "claude-opus-4-8"},
            "trigger": {"type": "refusal", "category": "cyber"},
        },
        {"type": "text", "text": "served by fallback"},
    ]
    rec = Recorder(
        httpx2.Response(200, json=message(model="claude-opus-4-8", usage=usage, content=content))
    )
    out = run(anthropic_provider(rec).complete(req()))
    assert out.text == "served by fallback"
    assert out.served_model == "claude-opus-4-8"
    assert [(u.model, u.output_tokens) for u in out.usage] == [
        ("claude-opus-5-5", 0),
        ("claude-opus-4-8", 40),
    ]


def test_anthropic_refusal_is_a_response_not_an_error() -> None:
    rec = Recorder(
        httpx2.Response(
            200,
            json=message(
                content=[],
                stop_reason="refusal",
                stop_details={"type": "refusal", "category": "cyber", "explanation": None},
            ),
        )
    )
    out = run(anthropic_provider(rec).complete(req()))
    assert (out.text, out.stop_reason) == ("", "refusal")
    assert out.extra == {"refusal_category": "cyber"}


def _error(status: int, kind: str, headers: dict[str, str] | None = None) -> httpx2.Response:
    return httpx2.Response(
        status,
        headers=headers or {},
        json={"type": "error", "error": {"type": kind, "message": "nope"}},
    )


@pytest.mark.parametrize(
    ("status", "kind", "retryable"),
    [
        (429, "rate_limit_error", True),
        (500, "api_error", True),
        (529, "overloaded_error", True),
        (400, "invalid_request_error", False),
        (404, "not_found_error", False),
    ],
)
def test_anthropic_error_classification(status: int, kind: str, retryable: bool) -> None:
    rec = Recorder(_error(status, kind, {"retry-after": "7"}))
    with pytest.raises(ProviderError) as info:
        run(anthropic_provider(rec).complete(req()))
    assert (info.value.status, info.value.retryable) == (status, retryable)
    assert info.value.retry_after_s == 7.0
    assert len(rec.requests) == 1  # the SDK's own retries are disabled


def test_anthropic_timeout_is_retryable() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ReadTimeout("slow", request=request)

    with pytest.raises(ProviderError) as info:
        run(anthropic_provider(handler).complete(req()))
    assert info.value.retryable is True
    assert "timed out" in str(info.value)


# --------------------------------------------------------------------- openai


def openai_provider(handler: Handler, **cfg: Any) -> OpenAICompatProvider:
    config = OpenAIProviderConfig.model_validate(
        {"type": "openai", "model": "local-model", "base_url": "http://127.0.0.1:9/v1", **cfg}
    )
    client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    return OpenAICompatProvider(config, client=client)


def chat_response(text: str = "hi", **usage: int) -> dict[str, Any]:
    return {
        "model": "local-model-2026-01",
        "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": usage.get("prompt", 20),
            "completion_tokens": usage.get("completion", 5),
            "prompt_tokens_details": {"cached_tokens": usage.get("cached", 8)},
        },
    }


def test_openai_request_shape_auth_and_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVALKIT_TEST_TOKEN", "placeholder-token")
    rec = Recorder(httpx2.Response(200, json=chat_response("answer")))
    p = openai_provider(
        rec,
        api_key_env="EVALKIT_TEST_TOKEN",
        temperature=0.7,
        seed=3,
        max_tokens_field="max_completion_tokens",
    )
    out = run(p.complete(req(schema=JUDGE_SCHEMA)))
    sent = rec.requests[0]
    assert str(sent.url) == "http://127.0.0.1:9/v1/chat/completions"
    assert sent.headers["authorization"] == "Bearer placeholder-token"
    body = rec.body()
    assert body["messages"][0] == {"role": "system", "content": "be terse"}
    assert body["max_completion_tokens"] == 1024
    assert (body["temperature"], body["seed"]) == (0.7, 3)
    assert "top_p" not in body
    assert body["response_format"]["json_schema"]["schema"] == JUDGE_SCHEMA
    assert out.text == "answer"
    assert out.served_model == "local-model-2026-01"
    (u,) = out.usage
    assert u.model == "local-model"  # priced under the configured name
    assert (u.input_tokens, u.cache_read_input_tokens, u.output_tokens) == (12, 8, 5)


def test_openai_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EVALKIT_MISSING_TOKEN", raising=False)
    with pytest.raises(MissingCredentialError, match="EVALKIT_MISSING_TOKEN"):
        openai_provider(Recorder(), api_key_env="EVALKIT_MISSING_TOKEN")
    rec = Recorder(httpx2.Response(200, json=chat_response()))
    run(openai_provider(rec, api_key_env=None).complete(req()))
    assert "authorization" not in rec.requests[0].headers


def test_openai_errors() -> None:
    rec = Recorder(httpx2.Response(429, headers={"retry-after": "2"}, text="slow down"))
    with pytest.raises(ProviderError) as info:
        run(openai_provider(rec, api_key_env=None).complete(req()))
    assert (info.value.retryable, info.value.retry_after_s) == (True, 2.0)

    rec = Recorder(httpx2.Response(400, text="bad"))
    with pytest.raises(ProviderError) as info:
        run(openai_provider(rec, api_key_env=None).complete(req()))
    assert info.value.retryable is False

    rec = Recorder(httpx2.Response(200, json={"choices": []}))
    with pytest.raises(ProviderError, match="malformed"):
        run(openai_provider(rec, api_key_env=None).complete(req()))

    def boom(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("refused", request=request)

    with pytest.raises(ProviderError) as info:
        run(openai_provider(boom, api_key_env=None).complete(req()))
    assert info.value.retryable is True
