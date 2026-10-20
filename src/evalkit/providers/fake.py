"""Deterministic scripted provider for tests, demos and CI dry-runs.

A script is a list of rules matched top to bottom against the flattened prompt
(system + messages). The first matching rule answers; its responses are cycled by
sample index, so trial *i* of a case always gets the same answer and a rule with
nine good responses and one bad one behaves like a 90%-reliable model.

`fail_attempts` makes the first attempts of every sample raise the listed HTTP
statuses, which exercises the retry path without a network.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from evalkit.providers.base import (
    Completion,
    CompletionRequest,
    Provider,
    ProviderError,
    TokenUsage,
    is_retryable_status,
)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FakeResponse(_Strict):
    text: str = ""
    latency_ms: float = Field(default=25.0, ge=0)
    stop_reason: str = "end_turn"
    repeat: int = Field(default=1, ge=1, le=10_000)


class FakeRule(_Strict):
    when: str | None = None
    when_regex: str | None = None
    model: str | None = None
    responses: list[str | FakeResponse] = Field(min_length=1)
    fail_attempts: list[int] = Field(default_factory=list)

    @field_validator("when_regex")
    @classmethod
    def _compiles(cls, value: str | None) -> str | None:
        if value is not None:
            try:
                re.compile(value)
            except re.error as exc:
                raise ValueError(f"invalid regex: {exc}") from exc
        return value

    @field_validator("fail_attempts")
    @classmethod
    def _http_statuses(cls, value: list[int]) -> list[int]:
        for status in value:
            if not 400 <= status <= 599:
                raise ValueError(f"fail_attempts must be HTTP error statuses, got {status}")
        return value

    @model_validator(mode="after")
    def _one_matcher(self) -> FakeRule:
        if self.when is not None and self.when_regex is not None:
            raise ValueError("use either `when` or `when_regex`, not both")
        return self

    def matches(self, prompt: str, model: str) -> bool:
        if self.model is not None and self.model != model:
            return False
        if self.when is not None:
            return self.when in prompt
        if self.when_regex is not None:
            return re.search(self.when_regex, prompt) is not None
        return True

    def expanded(self) -> list[FakeResponse]:
        out: list[FakeResponse] = []
        for item in self.responses:
            resp = FakeResponse(text=item) if isinstance(item, str) else item
            out.extend([resp] * resp.repeat)
        return out


class FakeScript(_Strict):
    rules: list[FakeRule] = Field(default_factory=list)
    default: str | FakeResponse | None = None

    def fingerprint(self) -> str:
        blob = json.dumps(self.model_dump(mode="json"), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()


def estimate_tokens(text: str) -> int:
    """~4 characters per token. Only used by the fake provider."""
    return max(1, math.ceil(len(text) / 4))


class FakeProvider(Provider):
    name = "fake"

    def __init__(self, model: str, script: FakeScript) -> None:
        super().__init__(model)
        self.script = script
        self._expanded = [rule.expanded() for rule in script.rules]
        self._attempts: dict[tuple[str, int], int] = {}
        self._fingerprint = script.fingerprint()
        self.calls = 0

    def identity(self) -> dict[str, Any]:
        return {"type": self.name, "model": self.model, "script": self._fingerprint}

    def _select(self, prompt: str) -> tuple[FakeRule | None, list[FakeResponse]]:
        for rule, responses in zip(self.script.rules, self._expanded, strict=True):
            if rule.matches(prompt, self.model):
                return rule, responses
        default = self.script.default
        if default is None:
            return None, []
        return None, [FakeResponse(text=default) if isinstance(default, str) else default]

    async def complete(self, request: CompletionRequest) -> Completion:
        self.calls += 1
        await asyncio.sleep(0)  # behave like a real awaitable call
        prompt = request.prompt_text()
        rule, responses = self._select(prompt)
        if not responses:
            raise ProviderError(
                "fake provider: no rule matched and the script has no default", retryable=False
            )
        if rule is not None and rule.fail_attempts:
            key = (hashlib.sha256(prompt.encode()).hexdigest(), request.sample_index)
            attempt = self._attempts.get(key, 0)
            self._attempts[key] = attempt + 1
            if attempt < len(rule.fail_attempts):
                status = rule.fail_attempts[attempt]
                raise ProviderError(
                    "fake provider: scripted failure",
                    retryable=is_retryable_status(status),
                    status=status,
                )
        resp = responses[request.sample_index % len(responses)]
        usage = TokenUsage(
            model=self.model,
            input_tokens=estimate_tokens(prompt),
            output_tokens=estimate_tokens(resp.text),
        )
        return Completion(
            text=resp.text,
            latency_ms=resp.latency_ms,
            usage=(usage,),
            served_model=self.model,
            stop_reason=resp.stop_reason,
        )
