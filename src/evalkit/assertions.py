"""Deterministic assertions over one model output.

`llm_judge` is asynchronous (it calls a model) and lives in judge.py; everything
here is a pure function of the trial.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from jsonschema.validators import validator_for

from evalkit.config import (
    Assertion,
    Contains,
    Equals,
    JsonSchema,
    LlmJudge,
    MaxCostUsd,
    MaxLatencyMs,
    NumericClose,
    Regex,
    fmt_num,
)


@dataclass(frozen=True)
class AssertionResult:
    name: str
    type: str
    passed: bool
    detail: str = ""
    gating: bool = True  # False: reported, but does not decide pass/fail
    score: float | None = None
    fingerprint: str | None = None  # llm_judge only: identifies judge + rubric

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return {k: v for k, v in data.items() if v is not None}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AssertionResult:
        return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})


@dataclass(frozen=True)
class TrialContext:
    output: str
    latency_ms: float
    cost_usd: float | None
    strip_code_fences: bool = False


class OutputError(ValueError):
    """The output (or a path inside it) is not what the assertion needs."""


_FENCE = re.compile(r"^\s*```[A-Za-z0-9_-]*\s*\n(.*?)\n?```\s*$", re.DOTALL)


def parse_json_output(text: str, *, strip_code_fences: bool = False) -> Any:
    candidate = text
    if strip_code_fences:
        match = _FENCE.match(text)
        if match:
            candidate = match.group(1)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise OutputError(f"output is not valid JSON ({exc.msg} at char {exc.pos})") from exc


def resolve_path(data: Any, path: str) -> Any:
    """Dotted path with integer list indexes: `items.0.price`."""
    node = data
    for part in path.split("."):
        if isinstance(node, list):
            if not part.lstrip("-").isdigit():
                raise OutputError(f"path {path!r}: expected a list index, got {part!r}")
            idx = int(part)
            if not -len(node) <= idx < len(node):
                raise OutputError(f"path {path!r}: index {idx} out of range")
            node = node[idx]
        elif isinstance(node, dict):
            if part not in node:
                raise OutputError(f"path {path!r}: key {part!r} not found")
            node = node[part]
        else:
            raise OutputError(f"path {path!r}: cannot index into {type(node).__name__}")
    return node


def _target(ctx: TrialContext, path: str | None) -> Any:
    if path is None:
        return ctx.output
    return resolve_path(
        parse_json_output(ctx.output, strip_code_fences=ctx.strip_code_fences), path
    )


def _as_text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _preview(value: Any, limit: int = 60) -> str:
    text = " ".join(_as_text(value).split())
    return repr(text if len(text) <= limit else text[: limit - 3] + "...")


def _equals(a: Equals, ctx: TrialContext) -> tuple[bool, str]:
    got = _target(ctx, a.path)
    if a.path is not None and not isinstance(got, str):
        ok = bool(got == a.value)
        return ok, "" if ok else f"got {_preview(got)}, expected {_preview(a.value)}"
    got_s, want_s = _as_text(got), _as_text(a.value)
    if a.strip:
        got_s, want_s = got_s.strip(), want_s.strip()
    if a.ignore_case:
        got_s, want_s = got_s.casefold(), want_s.casefold()
    ok = got_s == want_s
    return ok, "" if ok else f"got {_preview(got_s)}, expected {_preview(want_s)}"


def _contains(a: Contains, ctx: TrialContext) -> tuple[bool, str]:
    hay = _as_text(_target(ctx, a.path))
    needle = a.value
    if a.ignore_case:
        hay, needle = hay.casefold(), needle.casefold()
    found = needle in hay
    if a.negate:
        return (not found), (f"unexpected {a.value!r} found" if found else "")
    return found, "" if found else f"{a.value!r} not found in {_preview(hay)}"


def _regex(a: Regex, ctx: TrialContext) -> tuple[bool, str]:
    text = _as_text(_target(ctx, a.path))
    flags = re.IGNORECASE if a.ignore_case else 0
    fn = re.fullmatch if a.fullmatch else re.search
    found = fn(a.pattern, text, flags) is not None
    if a.negate:
        return (not found), ("pattern matched" if found else "")
    return found, "" if found else f"no match in {_preview(text)}"


_VALIDATORS: dict[int, tuple[dict[str, Any], Any]] = {}


def _validator(schema: dict[str, Any]) -> Any:
    """Compile each schema once per process (keyed by identity; schemas are immutable config)."""
    entry = _VALIDATORS.get(id(schema))
    if entry is None or entry[0] is not schema:
        entry = (schema, validator_for(schema)(schema))
        _VALIDATORS[id(schema)] = entry
    return entry[1]


def _json_schema(a: JsonSchema, ctx: TrialContext) -> tuple[bool, str]:
    data = parse_json_output(ctx.output, strip_code_fences=ctx.strip_code_fences)
    validator = _validator(a.schema_)
    errors = sorted(validator.iter_errors(data), key=lambda e: list(e.absolute_path))
    if not errors:
        return True, ""
    first = errors[0]
    where = "/".join(str(p) for p in first.absolute_path) or "(root)"
    more = f" (+{len(errors) - 1} more)" if len(errors) > 1 else ""
    return False, f"{where}: {first.message}{more}"


def _numeric_close(a: NumericClose, ctx: TrialContext) -> tuple[bool, str]:
    raw = _target(ctx, a.path)
    if isinstance(raw, bool):
        raise OutputError(f"expected a number, got {raw!r}")
    if isinstance(raw, int | float):
        value = float(raw)
    else:
        try:
            value = float(str(raw).strip())
        except ValueError as exc:
            raise OutputError(f"expected a number, got {_preview(raw)}") from exc
    if not math.isfinite(value):
        return False, f"got non-finite {value}"
    tolerance = max(a.abs_tol, a.rel_tol * abs(a.expected))
    ok = abs(value - a.expected) <= tolerance
    return (
        ok,
        ""
        if ok
        else f"got {fmt_num(value)}, expected {fmt_num(a.expected)} +/- {fmt_num(tolerance)}",
    )


def evaluate(assertion: Assertion, ctx: TrialContext) -> AssertionResult:
    """Evaluate a deterministic assertion. `llm_judge` is handled by judge.py."""
    if isinstance(assertion, LlmJudge):
        raise TypeError("llm_judge is evaluated asynchronously by evalkit.judge")
    try:
        if isinstance(assertion, Equals):
            ok, detail = _equals(assertion, ctx)
        elif isinstance(assertion, Contains):
            ok, detail = _contains(assertion, ctx)
        elif isinstance(assertion, Regex):
            ok, detail = _regex(assertion, ctx)
        elif isinstance(assertion, JsonSchema):
            ok, detail = _json_schema(assertion, ctx)
        elif isinstance(assertion, NumericClose):
            ok, detail = _numeric_close(assertion, ctx)
        elif isinstance(assertion, MaxLatencyMs):
            ok = ctx.latency_ms <= assertion.value
            detail = "" if ok else f"{ctx.latency_ms:.0f} ms > {fmt_num(assertion.value)} ms"
        elif isinstance(assertion, MaxCostUsd):
            if ctx.cost_usd is None:
                ok, detail = False, "cost unknown (no price for the model that served it)"
            else:
                ok = ctx.cost_usd <= assertion.value
                detail = "" if ok else f"${ctx.cost_usd:.6f} > ${fmt_num(assertion.value)}"
        else:  # pragma: no cover - the union is exhaustive
            raise TypeError(f"unknown assertion {assertion!r}")
    except OutputError as exc:
        ok, detail = False, str(exc)
    return AssertionResult(
        name=assertion.display_name(), type=assertion.type, passed=ok, detail=detail
    )
