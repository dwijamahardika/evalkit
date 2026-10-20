from __future__ import annotations

from typing import Any

import pytest
from pydantic import TypeAdapter

from evalkit import judge
from evalkit.assertions import (
    OutputError,
    TrialContext,
    evaluate,
    parse_json_output,
    resolve_path,
)
from evalkit.config import Assertion, CalibrationFile, JudgeConfig, LlmJudge

_ADAPTER: TypeAdapter[Any] = TypeAdapter(Assertion)


def check(spec: dict[str, Any], output: str, **ctx: Any) -> tuple[bool, str]:
    assertion = _ADAPTER.validate_python(spec)
    context = TrialContext(
        output=output,
        latency_ms=ctx.get("latency_ms", 100.0),
        cost_usd=ctx.get("cost_usd", 0.001),
        strip_code_fences=ctx.get("strip_code_fences", False),
    )
    result = evaluate(assertion, context)
    return result.passed, result.detail


RECEIPT = (
    '{"total": 45500, "payment_method": "cash", '
    '"items": [{"name": "Roti", "qty": 1.5}], "ok": true}'
)


@pytest.mark.parametrize(
    ("spec", "output", "expected"),
    [
        ({"type": "equals", "value": "yes"}, "  yes \n", True),
        ({"type": "equals", "value": "YES", "ignore_case": True}, "yes", True),
        ({"type": "equals", "value": "yes", "strip": False}, " yes", False),
        ({"type": "equals", "path": "total", "value": 45500}, RECEIPT, True),
        ({"type": "equals", "path": "total", "value": 45000}, RECEIPT, False),
        (
            {"type": "equals", "path": "payment_method", "value": "CASH", "ignore_case": True},
            RECEIPT,
            True,
        ),
        ({"type": "contains", "value": "Rp"}, "Total Rp 45.500", True),
        ({"type": "contains", "value": "rp", "ignore_case": True}, "Total Rp 45.500", True),
        ({"type": "contains", "value": "```", "negate": True}, "plain", True),
        ({"type": "contains", "value": "```", "negate": True}, "```json", False),
        ({"type": "contains", "path": "items.0.name", "value": "Rot"}, RECEIPT, True),
        ({"type": "regex", "pattern": r"Rp\s?\d{1,3}(\.\d{3})+"}, "Total Rp 45.500", True),
        ({"type": "regex", "pattern": r"\d+", "fullmatch": True}, "123a", False),
        ({"type": "regex", "pattern": "TOTAL", "ignore_case": True}, "total", True),
        ({"type": "regex", "pattern": "error", "negate": True}, "fine", True),
        ({"type": "numeric_close", "path": "total", "expected": 45500}, RECEIPT, True),
        (
            {"type": "numeric_close", "path": "total", "expected": 45000, "abs_tol": 499},
            RECEIPT,
            False,
        ),
        (
            {"type": "numeric_close", "path": "total", "expected": 45000, "rel_tol": 0.02},
            RECEIPT,
            True,
        ),
        ({"type": "numeric_close", "path": "items.0.qty", "expected": 1.5}, RECEIPT, True),
        ({"type": "numeric_close", "expected": 3.14, "abs_tol": 0.01}, " 3.141 ", True),
        (
            {"type": "numeric_close", "path": "ok", "expected": 1},
            RECEIPT,
            False,
        ),  # bool is not a number
        ({"type": "numeric_close", "expected": 1}, "one", False),
        ({"type": "max_latency_ms", "value": 100}, "x", True),
    ],
)
def test_deterministic_assertions(spec: dict[str, Any], output: str, expected: bool) -> None:
    passed, detail = check(spec, output)
    assert passed is expected, detail
    assert bool(detail) is (not expected)


def test_latency_and_cost() -> None:
    assert check({"type": "max_latency_ms", "value": 50}, "x", latency_ms=51.0)[0] is False
    assert check({"type": "max_cost_usd", "value": 0.01}, "x", cost_usd=0.009)[0] is True
    passed, detail = check({"type": "max_cost_usd", "value": 0.01}, "x", cost_usd=None)
    assert passed is False
    assert "cost unknown" in detail


SCHEMA = {
    "type": "object",
    "required": ["total"],
    "properties": {"total": {"type": "integer"}, "items": {"type": "array"}},
}


def test_json_schema() -> None:
    assert check({"type": "json_schema", "schema": SCHEMA}, '{"total": 1}')[0] is True
    passed, detail = check({"type": "json_schema", "schema": SCHEMA}, '{"total": "1", "items": 3}')
    assert passed is False
    assert detail.startswith("items: 3 is not of type 'array' (+1 more)")
    passed, detail = check({"type": "json_schema", "schema": SCHEMA}, "not json")
    assert passed is False
    assert "not valid JSON" in detail


def test_code_fences_only_stripped_when_enabled() -> None:
    fenced = '```json\n{"total": 1}\n```'
    assert check({"type": "json_schema", "schema": SCHEMA}, fenced)[0] is False
    assert check({"type": "json_schema", "schema": SCHEMA}, fenced, strip_code_fences=True)[0]
    assert parse_json_output("```\n[1]\n```", strip_code_fences=True) == [1]


def test_resolve_path_errors() -> None:
    data = {"a": [{"b": 1}]}
    assert resolve_path(data, "a.0.b") == 1
    assert resolve_path(data, "a.-1.b") == 1
    for path, msg in [
        ("a.x", "expected a list index"),
        ("a.5", "out of range"),
        ("c", "not found"),
        ("a.0.b.c", "cannot index"),
    ]:
        with pytest.raises(OutputError, match=msg):
            resolve_path(data, path)


def test_llm_judge_is_not_evaluated_synchronously() -> None:
    with pytest.raises(TypeError):
        evaluate(LlmJudge(type="llm_judge", rubric="r"), TrialContext("x", 1.0, None))


# ---------------------------------------------------------------------- judge


@pytest.mark.parametrize(
    ("text", "score"),
    [
        ('{"reason": "fine", "score": 5}', 5),
        ('Here you go:\n```json\n{"score": 2, "reason": "invented item"}\n```', 2),
    ],
)
def test_parse_verdict(text: str, score: int) -> None:
    assert judge.parse_verdict(text)[0] == score


@pytest.mark.parametrize(
    "text", ["no json here", '{"score": 9}', '{"score": true}', '{"score": "5"}', "{broken"]
)
def test_parse_verdict_rejects(text: str) -> None:
    with pytest.raises(judge.JudgeParseError):
        judge.parse_verdict(text)


def test_fingerprint_tracks_what_matters() -> None:
    a = LlmJudge(type="llm_judge", rubric="be faithful", threshold=4)
    ident = {"type": "fake", "model": "j"}
    fp = judge.fingerprint(ident, a)
    assert fp == judge.fingerprint(dict(reversed(list(ident.items()))), a)
    assert fp == judge.fingerprint(ident, a.model_copy(update={"name": "renamed"}))
    assert fp != judge.fingerprint(ident, a.model_copy(update={"rubric": "be kind"}))
    assert fp != judge.fingerprint(ident, a.model_copy(update={"threshold": 3}))
    assert fp != judge.fingerprint({**ident, "model": "j2"}, a)


def _judge_cfg(min_kappa: float | None) -> JudgeConfig:
    return JudgeConfig.model_validate(
        {
            "provider": {"type": "fake", "model": "j", "script": {"rules": []}},
            "min_kappa": min_kappa,
        }
    )


def _calibration(kappa: float | None) -> CalibrationFile:
    return CalibrationFile.model_validate(
        {
            "schema_version": 1,
            "judges": {"fp": {"name": "j", "n": 40, "agreement": 0.9, "kappa": kappa}},
        }
    )


def test_gating_decisions() -> None:
    assert judge.decide_gating(_judge_cfg(None), None, "fp", "j").gating is True
    missing = judge.decide_gating(_judge_cfg(0.6), None, "fp", "j")
    assert (missing.gating, missing.note) == (False, "advisory: no calibration file")
    other = judge.decide_gating(_judge_cfg(0.6), _calibration(0.9), "other-fp", "j")
    assert other.gating is False
    assert "no calibration for this rubric" in other.note
    low = judge.decide_gating(_judge_cfg(0.6), _calibration(0.41), "fp", "j")
    assert (low.gating, low.kappa) == (False, 0.41)
    assert judge.decide_gating(_judge_cfg(0.6), _calibration(None), "fp", "j").gating is False
    ok = judge.decide_gating(_judge_cfg(0.6), _calibration(0.75), "fp", "j")
    assert ok.gating is True
    assert ok.note == "gating: kappa 0.75 >= 0.60 (n=40)"


def test_judge_request_carries_schema_and_sample() -> None:
    req = judge.build_request("task", "output", "rubric", sample_index=7)
    assert req.sample_index == 7
    assert req.response_schema == judge.JUDGE_SCHEMA
    assert "<rubric>\nrubric\n</rubric>" in req.messages[0].content
