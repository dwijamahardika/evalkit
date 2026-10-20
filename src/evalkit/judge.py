"""LLM-as-judge: rubric-scored assertions, and when to trust them.

A judge is a model, so it has an error rate. evalkit fingerprints every judge
(provider identity + rubric + threshold + prompt version); `evalkit calibrate`
measures agreement with human labels per fingerprint, and `judge.min_kappa`
turns an uncalibrated or poorly-agreeing judge into an advisory signal instead
of a CI gate. Changing the rubric or the judge model invalidates the
calibration automatically, because the fingerprint changes.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evalkit.config import CalibrationFile, JudgeConfig, LlmJudge
from evalkit.providers.base import ChatMessage, CompletionRequest

JUDGE_PROMPT_VERSION = 1

JUDGE_SYSTEM = (
    "You are a strict evaluator inside an automated test suite. You receive the "
    "task given to an assistant, the assistant's response, and a rubric. Grade the "
    "response against the rubric only.\n\n"
    "Scale: 5 = fully meets the rubric; 4 = meets it with trivial issues; "
    "3 = partially meets it; 2 = mostly fails it; 1 = completely fails it.\n\n"
    'Reply with a single JSON object and nothing else: {"reason": "<one short '
    'sentence>", "score": <integer 1-5>}'
)

JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"reason": {"type": "string"}, "score": {"type": "integer"}},
    "required": ["reason", "score"],
    "additionalProperties": False,
}


class JudgeParseError(ValueError):
    pass


def fingerprint(judge_identity: dict[str, Any], assertion: LlmJudge) -> str:
    payload = {
        "v": JUDGE_PROMPT_VERSION,
        "judge": judge_identity,
        "rubric": assertion.rubric,  # the template, so every case shares one calibration
        "threshold": assertion.threshold,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def build_request(task: str, output: str, rubric: str, sample_index: int) -> CompletionRequest:
    user = (
        f"<task>\n{task}\n</task>\n\n"
        f"<response>\n{output}\n</response>\n\n"
        f"<rubric>\n{rubric}\n</rubric>"
    )
    return CompletionRequest(
        system=JUDGE_SYSTEM,
        messages=(ChatMessage("user", user),),
        sample_index=sample_index,
        response_schema=JUDGE_SCHEMA,
    )


_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def parse_verdict(text: str) -> tuple[int, str]:
    match = _JSON_OBJECT.search(text)
    if not match:
        raise JudgeParseError(f"judge reply has no JSON object: {text[:80]!r}")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise JudgeParseError(f"judge reply is not valid JSON: {exc.msg}") from exc
    score = data.get("score") if isinstance(data, dict) else None
    if isinstance(score, bool) or not isinstance(score, int) or not 1 <= score <= 5:
        raise JudgeParseError(f"judge score must be an integer 1-5, got {score!r}")
    reason = data.get("reason", "")
    return score, reason if isinstance(reason, str) else str(reason)


@dataclass(frozen=True)
class GatingDecision:
    fingerprint: str
    name: str
    gating: bool
    kappa: float | None
    note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "fingerprint": self.fingerprint,
            "name": self.name,
            "gating": self.gating,
            "kappa": self.kappa,
            "note": self.note,
        }


def load_calibration(judge: JudgeConfig, base_dir: Path) -> CalibrationFile | None:
    if judge.calibration is None:
        return None
    path = base_dir / judge.calibration
    if not path.is_file():
        return None
    return CalibrationFile.model_validate_json(path.read_text(encoding="utf-8"))


def decide_gating(
    judge: JudgeConfig, calibration: CalibrationFile | None, fp: str, name: str
) -> GatingDecision:
    if judge.min_kappa is None:
        return GatingDecision(fp, name, True, None, "gating (no min_kappa configured)")
    entry = calibration.judges.get(fp) if calibration else None
    if entry is None:
        why = (
            "no calibration file" if calibration is None else "no calibration for this rubric/model"
        )
        return GatingDecision(fp, name, False, None, f"advisory: {why}")
    if entry.kappa is None or entry.kappa < judge.min_kappa:
        shown = "undefined" if entry.kappa is None else f"{entry.kappa:.2f}"
        return GatingDecision(
            fp,
            name,
            False,
            entry.kappa,
            f"advisory: kappa {shown} < min_kappa {judge.min_kappa:.2f}",
        )
    return GatingDecision(
        fp,
        name,
        True,
        entry.kappa,
        f"gating: kappa {entry.kappa:.2f} >= {judge.min_kappa:.2f} (n={entry.n})",
    )
