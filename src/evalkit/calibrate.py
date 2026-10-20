"""Measure how far an LLM judge can be trusted, against human labels.

Input: a run containing llm_judge results, and a CSV with columns
`case_id,trial,label[,assertion]` (label: pass/fail, true/false, yes/no, 1/0).
Output: per-judge agreement, Cohen's kappa and a confusion matrix, optionally
written as the calibration file that `judge.calibration` points to.

Agreement alone flatters a judge on a lopsided suite (if 90% of outputs are good,
"always pass" agrees 90% of the time); kappa corrects for chance agreement.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evalkit.assertions import AssertionResult
from evalkit.runstore import Run
from evalkit.stats import Agreement, cohen_kappa, kappa_label

_TRUE = {"pass", "passed", "true", "yes", "y", "1"}
_FALSE = {"fail", "failed", "false", "no", "n", "0"}


class CalibrationError(ValueError):
    pass


@dataclass(frozen=True)
class HumanLabel:
    case_id: str
    trial: int
    passed: bool
    assertion: str | None = None


def read_labels(path: Path) -> list[HumanLabel]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CalibrationError(f"{path}: cannot read: {exc.strerror}") from exc
    # Drop `#` comment lines, remembering each kept line's number in the file.
    kept = [
        (n, line)
        for n, line in enumerate(text.splitlines(), start=1)
        if not line.lstrip().startswith("#")
    ]
    reader = csv.DictReader([line for _, line in kept])
    required = {"case_id", "trial", "label"}
    missing = required - set(reader.fieldnames or [])
    if missing:
        raise CalibrationError(f"{path}: missing column(s): {', '.join(sorted(missing))}")
    labels: list[HumanLabel] = []
    seen: set[tuple[str, int, str | None]] = set()
    problems: list[str] = []
    for row in reader:
        lineno = kept[reader.line_num - 1][0]
        case_id = (row.get("case_id") or "").strip()
        raw_label = (row.get("label") or "").strip().casefold()
        assertion = (row.get("assertion") or "").strip() or None
        try:
            trial = int((row.get("trial") or "").strip())
        except ValueError:
            problems.append(f"line {lineno}: trial must be an integer")
            continue
        if raw_label in _TRUE:
            passed = True
        elif raw_label in _FALSE:
            passed = False
        else:
            problems.append(f"line {lineno}: label {raw_label!r} is not pass/fail")
            continue
        key = (case_id, trial, assertion)
        if key in seen:
            problems.append(f"line {lineno}: duplicate label for {case_id} trial {trial}")
            continue
        seen.add(key)
        labels.append(HumanLabel(case_id, trial, passed, assertion))
    if problems:
        raise CalibrationError(f"{path}:\n  " + "\n  ".join(problems))
    return labels


@dataclass(frozen=True)
class JudgeAgreement:
    fingerprint: str
    name: str
    agreement: Agreement


@dataclass(frozen=True)
class CalibrationResult:
    run_id: str
    suite: str
    judge: dict[str, Any] | None
    judges: list[JudgeAgreement]
    overall: Agreement | None
    unmatched: list[HumanLabel] = field(default_factory=list)
    unscored: int = 0  # labelled trials whose judge produced no score

    def worst_kappa(self) -> float | None:
        kappas = [j.agreement.kappa for j in self.judges]
        if not kappas or any(k is None for k in kappas):
            return None
        return min(k for k in kappas if k is not None)

    def to_file_dict(self) -> dict[str, Any]:
        def block(name: str, a: Agreement) -> dict[str, Any]:
            return {
                "name": name,
                "n": a.n,
                "agreement": round(a.observed, 6),
                "kappa": None if a.kappa is None else round(a.kappa, 6),
                "kappa_label": kappa_label(a.kappa),
                "confusion": {
                    f"judge={jl}": {f"human={hl}": n for hl, n in row.items()}
                    for jl, row in a.confusion.items()
                },
            }

        return {
            "schema_version": 1,
            "created_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            "suite": self.suite,
            "run_id": self.run_id,
            "judge": self.judge,
            "judges": {j.fingerprint: block(j.name, j.agreement) for j in self.judges},
            "overall": None if self.overall is None else block("overall", self.overall),
            "unmatched_labels": len(self.unmatched),
        }


def calibrate(run: Run, labels: list[HumanLabel]) -> CalibrationResult:
    index: dict[tuple[str, int], list[AssertionResult]] = defaultdict(list)
    for trial in run.trials:
        for result in trial.assertions:
            if result.type == "llm_judge":
                index[(trial.case_id, trial.trial)].append(result)
    if not index:
        raise CalibrationError("the run contains no llm_judge results")

    pairs: dict[str, list[tuple[str, str]]] = defaultdict(list)
    names: dict[str, str] = {}
    unmatched: list[HumanLabel] = []
    unscored = 0
    for label in labels:
        candidates = index.get((label.case_id, label.trial), [])
        if label.assertion is not None:
            candidates = [r for r in candidates if r.name == label.assertion]
        if not candidates:
            unmatched.append(label)
            continue
        if len(candidates) > 1:
            raise CalibrationError(
                f"{label.case_id} trial {label.trial} has {len(candidates)} judge assertions; "
                "add an `assertion` column naming which one each label is for"
            )
        result = candidates[0]
        if result.score is None or result.fingerprint is None:
            unscored += 1
            continue
        names[result.fingerprint] = result.name
        judge_label = "pass" if result.passed else "fail"
        human_label = "pass" if label.passed else "fail"
        pairs[result.fingerprint].append((judge_label, human_label))

    judges = [
        JudgeAgreement(fp, names[fp], cohen_kappa(items)) for fp, items in sorted(pairs.items())
    ]
    every = [p for items in pairs.values() for p in items]
    return CalibrationResult(
        run_id=run.meta.run_id,
        suite=run.meta.suite,
        judge=run.meta.judge,
        judges=judges,
        overall=cohen_kappa(every) if every else None,
        unmatched=unmatched,
        unscored=unscored,
    )


def write_calibration(path: Path, result: CalibrationResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_file_dict(), indent=2) + "\n", encoding="utf-8")


def render_calibration_text(result: CalibrationResult) -> str:
    total = sum(j.agreement.n for j in result.judges)
    lines = [f"calibration: {total} labelled judgement(s) from run {result.run_id}"]
    for j in result.judges:
        a = j.agreement
        kappa = "undefined" if a.kappa is None else f"{a.kappa:.3f}"
        lines.append(
            f"  judge {j.name} [{j.fingerprint}]  n={a.n}  agreement {100 * a.observed:.1f}%  "
            f"kappa {kappa} ({kappa_label(a.kappa)})"
        )
        labels = list(a.confusion)
        lines.append("    " + " " * 12 + "".join(f"{'human=' + h:>13}" for h in labels))
        for jl in labels:
            row = "".join(f"{a.confusion[jl][h]:>13}" for h in labels)
            lines.append(f"    {'judge=' + jl:<12}{row}")
    if result.unmatched:
        lines.append(f"  {len(result.unmatched)} label(s) matched no judged trial in this run")
    if result.unscored:
        lines.append(f"  {result.unscored} labelled trial(s) had no judge score (judge errors)")
    return "\n".join(lines)
