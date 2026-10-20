from __future__ import annotations

import textwrap
from collections.abc import Callable
from pathlib import Path

import pytest

from evalkit.assertions import AssertionResult
from evalkit.runstore import CaseMeta, Run, RunMeta, TrialRecord

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_SUITE = ROOT / "examples" / "receipts" / "suite.yaml"
EXAMPLE_LABELS = ROOT / "examples" / "receipts" / "human_labels.csv"

MINIMAL_SCRIPT = """
rules:
  - when: "hello"
    responses: ["HELLO"]
default: "fallback"
"""


@pytest.fixture
def write_suite(tmp_path: Path) -> Callable[..., Path]:
    """Write a suite (plus side files) into tmp_path and return its path."""

    def _write(body: str, files: dict[str, str] | None = None, name: str = "suite.yaml") -> Path:
        for rel, content in (files or {}).items():
            target = tmp_path / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")
        path = tmp_path / name
        path.write_text(textwrap.dedent(body).lstrip(), encoding="utf-8")
        return path

    return _write


async def no_sleep(_: float) -> None:
    return None


def make_run(
    counts: dict[str, tuple[int, int]],
    *,
    model: str = "model-x",
    errors: dict[str, int] | None = None,
    min_pass_rate: float = 0.8,
    run_id: str = "run",
    suite: str = "suite",
    min_suite_pass_rate: float | None = None,
) -> Run:
    """A synthetic run: counts[case] = (passed, trials); errored trials come first."""
    errors = errors or {}
    trials: list[TrialRecord] = []
    for case_id, (passed, total) in counts.items():
        n_err = errors.get(case_id, 0)
        for i in range(total):
            errored = i < n_err
            ok = (not errored) and (i - n_err) < passed
            trials.append(
                TrialRecord(
                    case_id=case_id,
                    trial=i,
                    passed=ok,
                    output=None if errored else ("good" if ok else "bad"),
                    assertions=[]
                    if errored
                    else [AssertionResult(name="contains(good)", type="contains", passed=ok)],
                    latency_ms=None if errored else 100.0 + i,
                    usage={},
                    cost_usd=None if errored else 0.001,
                    served_model=None if errored else model,
                    error="boom (HTTP 503)" if errored else None,
                )
            )
    meta = RunMeta(
        run_id=run_id,
        suite=suite,
        suite_path="suite.yaml",
        suite_fingerprint="0" * 64,
        evalkit_version="test",
        started_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T00:00:01+00:00",
        provider={"type": "fake", "model": model},
        min_suite_pass_rate=min_suite_pass_rate,
        cases=[
            CaseMeta(id=c, trials=t, min_pass_rate=min_pass_rate) for c, (_, t) in counts.items()
        ],
    )
    return Run(meta=meta, trials=trials)
