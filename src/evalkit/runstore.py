"""Run files: JSON Lines, one header record then one record per trial.

A run file is self-describing: the header carries the gate thresholds, the case
list and the judge gating decisions, so `report` and `diff` need nothing but the
file (e.g. a CI artifact from another branch).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from evalkit.assertions import AssertionResult

RUN_SCHEMA_VERSION = 1


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CaseMeta(_Record):
    id: str
    trials: int
    min_pass_rate: float
    tags: list[str] = Field(default_factory=list)


class RunMeta(_Record):
    type: Literal["run"] = "run"
    schema_version: int = RUN_SCHEMA_VERSION
    run_id: str
    suite: str
    suite_path: str
    suite_fingerprint: str
    evalkit_version: str
    started_at: str
    finished_at: str
    provider: dict[str, Any]
    judge: dict[str, Any] | None = None
    min_suite_pass_rate: float | None = None
    cases: list[CaseMeta]
    judges: list[dict[str, Any]] = Field(default_factory=list)
    labels: dict[str, str] = Field(default_factory=dict)
    cache: dict[str, int] = Field(default_factory=dict)


class TrialRecord(_Record):
    type: Literal["trial"] = "trial"
    case_id: str
    trial: int
    passed: bool
    output: str | None
    assertions: list[AssertionResult] = Field(default_factory=list)
    latency_ms: float | None = None
    usage: dict[str, int] = Field(default_factory=dict)
    cost_usd: float | None = None
    judge_cost_usd: float | None = None
    cached: bool = False
    attempts: int = 0
    served_model: str | None = None
    stop_reason: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class Run:
    meta: RunMeta
    trials: list[TrialRecord]

    def by_case(self) -> dict[str, list[TrialRecord]]:
        grouped: dict[str, list[TrialRecord]] = {c.id: [] for c in self.meta.cases}
        for t in self.trials:
            grouped.setdefault(t.case_id, []).append(t)
        return grouped


class RunFormatError(ValueError):
    pass


def _dump(record: BaseModel) -> str:
    data = record.model_dump(mode="json")
    if isinstance(record, TrialRecord):
        data["assertions"] = [a.to_dict() for a in record.assertions]
    return json.dumps(data, ensure_ascii=False, sort_keys=False)


def write_run(path: Path, run: Run) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        fh.write(_dump(run.meta) + "\n")
        for trial in run.trials:
            fh.write(_dump(trial) + "\n")
    tmp.replace(path)


def read_run(path: Path) -> Run:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RunFormatError(f"{path}: cannot read: {exc.strerror}") from exc
    meta: RunMeta | None = None
    trials: list[TrialRecord] = []
    for lineno, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
            kind = data.get("type") if isinstance(data, dict) else None
            if kind == "run":
                if meta is not None:
                    raise RunFormatError(f"{path}:{lineno}: second run header")
                meta = RunMeta.model_validate(data)
                if meta.schema_version != RUN_SCHEMA_VERSION:
                    raise RunFormatError(
                        f"{path}: run schema v{meta.schema_version}, "
                        f"this evalkit reads v{RUN_SCHEMA_VERSION}"
                    )
            elif kind == "trial":
                trials.append(TrialRecord.model_validate(data))
            else:
                raise RunFormatError(f"{path}:{lineno}: unknown record type {kind!r}")
        except (json.JSONDecodeError, ValidationError) as exc:
            raise RunFormatError(f"{path}:{lineno}: {exc}") from exc
    if meta is None:
        raise RunFormatError(f"{path}: no run header (is this an evalkit run file?)")
    return Run(meta=meta, trials=trials)
