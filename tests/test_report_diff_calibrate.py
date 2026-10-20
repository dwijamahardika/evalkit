from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from evalkit.assertions import AssertionResult
from evalkit.calibrate import (
    CalibrationError,
    calibrate,
    read_labels,
    render_calibration_text,
    write_calibration,
)
from evalkit.config import CalibrationFile
from evalkit.diff import FailOn, diff_runs, render_diff_markdown, render_diff_text
from evalkit.report import (
    EXIT_GATE_FAILED,
    EXIT_INFRA,
    EXIT_OK,
    render_junit,
    render_markdown,
    render_text,
    summarize,
)
from evalkit.runstore import Run, TrialRecord
from tests.conftest import make_run

# --------------------------------------------------------------------- report


def test_gate_uses_observed_rate_with_float_tolerance() -> None:
    summary = summarize(make_run({"a": (8, 10), "b": (4, 5)}, min_pass_rate=0.8))
    assert all(c.gate_passed for c in summary.cases)
    assert summary.exit_code() == EXIT_OK
    failing = summarize(make_run({"a": (7, 10)}, min_pass_rate=0.8))
    assert failing.exit_code() == EXIT_GATE_FAILED
    suite_floor = summarize(make_run({"a": (8, 10)}, min_pass_rate=0.8, min_suite_pass_rate=0.9))
    assert suite_floor.cases[0].gate_passed
    assert suite_floor.exit_code() == EXIT_GATE_FAILED


def test_errors_take_precedence() -> None:
    summary = summarize(make_run({"a": (8, 10)}, errors={"a": 2}))
    assert summary.errors == 2
    assert summary.exit_code() == EXIT_INFRA
    assert summary.exit_code(allow_errors=True) == EXIT_OK
    assert "ERROR" in render_text(summary)


def test_renderers() -> None:
    run = make_run({"good": (10, 10), "bad|pipe": (2, 10)}, errors={"bad|pipe": 1})
    summary = summarize(run)
    text = render_text(summary)
    assert "result: ERROR" in text
    md = render_markdown(summary)
    assert "| PASS | `good` | 10/10 (100.0%) |" in md
    assert "`bad\\|pipe`" in md  # pipes in ids cannot break the table
    assert "<details><summary>Failure details</summary>" in md
    root = ET.fromstring(render_junit(summary).split("\n", 1)[1])
    suite = root.find("testsuite")
    assert suite is not None
    assert (suite.get("tests"), suite.get("failures"), suite.get("errors")) == ("2", "0", "1")
    cases = {tc.get("name"): tc for tc in suite.iter("testcase")}
    assert cases["good"].find("failure") is None
    assert cases["bad|pipe"].find("error") is not None


def test_advisory_failures_do_not_gate() -> None:
    run = make_run({"a": (10, 10)})
    trials = [
        t.model_copy(
            update={
                "assertions": [
                    *t.assertions,
                    AssertionResult("faithful", "llm_judge", passed=False, gating=False, score=2.0),
                ]
            }
        )
        for t in run.trials
    ]
    summary = summarize(Run(meta=run.meta, trials=trials))
    case = summary.cases[0]
    assert case.gate_passed
    assert case.advisory_failures == {"faithful": 10}
    assert "advisory: faithful x10" in render_text(summary)


# ----------------------------------------------------------------------- diff

EIGHT = {f"case-{i}": (10, 10) for i in range(8)}


def test_identical_runs_have_no_regressions() -> None:
    d = diff_runs(make_run(EIGHT), make_run(EIGHT))
    assert not d.regressions
    assert not d.improvements
    assert d.delta is not None
    assert (d.delta.low, d.delta.high) == (0.0, 0.0)
    assert d.exit_code() == EXIT_OK
    assert "OK (no significant regression)" in render_diff_text(d)


def test_clear_regression_vs_noise() -> None:
    base = make_run({**EIGHT, "flaky": (9, 10)})
    cand = make_run({**EIGHT, "case-3": (0, 10), "flaky": (7, 10)})
    d = diff_runs(base, cand)
    assert [c.id for c in d.regressions] == ["case-3"]
    flaky = next(c for c in d.cases if c.id == "flaky")
    assert flaky.status == "unchanged"
    assert flaky.delta == pytest.approx(-0.2)
    assert d.exit_code(FailOn.ANY) == EXIT_GATE_FAILED
    assert d.exit_code(FailOn.CASE) == EXIT_GATE_FAILED
    assert d.exit_code(FailOn.NONE) == EXIT_OK
    # One bad case among nine: the suite-level bootstrap cannot call it.
    assert d.suite_status == "no significant change"
    assert d.exit_code(FailOn.SUITE) == EXIT_OK
    md = render_diff_markdown(d)
    assert "| REGRESSED | `case-3` | 10/10 | 0/10 | -100.0 pp |" in md
    assert "1 case(s) moved but not significantly" in md


def test_multiple_comparison_correction_matters() -> None:
    # 10/10 -> 4/10 is p ~= 0.011 on its own; with 20 cases, Holm says noise.
    many = {f"c{i}": (10, 10) for i in range(20)}
    d = diff_runs(make_run(many), make_run({**many, "c0": (4, 10)}))
    c0 = d.cases[0]
    assert c0.p_value is not None
    assert c0.p_value < 0.05
    assert c0.p_adjusted is not None
    assert c0.p_adjusted > 0.05
    assert c0.status == "unchanged"
    # The same drop on a 1-case suite is flagged.
    alone = diff_runs(make_run({"c0": (10, 10)}), make_run({"c0": (4, 10)}))
    assert alone.cases[0].status == "regressed"


def test_suite_level_regression_detected() -> None:
    base = {f"c{i}": (9, 10) for i in range(12)}
    cand = {f"c{i}": (7, 10) for i in range(12)}
    d = diff_runs(make_run(base), make_run(cand))
    assert not d.regressions  # each case alone is noise...
    assert d.suite_status == "regressed"  # ...but all twelve moved the same way
    assert d.exit_code(FailOn.SUITE) == EXIT_GATE_FAILED
    assert d.exit_code(FailOn.CASE) == EXIT_OK


def test_added_removed_and_errors() -> None:
    base = make_run({"a": (5, 5), "old": (5, 5)})
    cand = make_run({"a": (5, 5), "new": (1, 5)}, errors={"a": 1})
    d = diff_runs(base, cand)
    status = {c.id: c.status for c in d.cases}
    assert status == {"a": "unchanged", "old": "removed", "new": "added"}
    assert any("only in base: old" in w for w in d.warnings)
    assert any("errored trials" in w for w in d.warnings)
    assert d.exit_code() == EXIT_INFRA
    assert d.exit_code(allow_errors=True) == EXIT_OK
    assert "warning:" in render_diff_text(d)


def test_power_line() -> None:
    d = diff_runs(make_run({"a": (3, 3)}), make_run({"a": (0, 3)}))
    assert d.detectable_drop is None
    assert "no single-case drop can reach significance" in render_diff_text(d)
    with pytest.raises(ValueError, match="alpha"):
        diff_runs(make_run({"a": (3, 3)}), make_run({"a": (3, 3)}), alpha=0)


# ------------------------------------------------------------------ calibrate


def judged_run(verdicts: dict[tuple[str, int], bool], fingerprint: str = "fp1") -> Run:
    cases: dict[str, tuple[int, int]] = {}
    for case_id, trial in verdicts:
        cases[case_id] = (0, max(cases.get(case_id, (0, 0))[1], trial + 1))
    run = make_run(cases)
    trials: list[TrialRecord] = []
    for t in run.trials:
        passed = verdicts.get((t.case_id, t.trial))
        if passed is None:
            trials.append(t)
            continue
        judge = AssertionResult(
            "faithful",
            "llm_judge",
            passed=passed,
            score=5.0 if passed else 2.0,
            fingerprint=fingerprint,
        )
        trials.append(t.model_copy(update={"assertions": [judge]}))
    return Run(meta=run.meta, trials=trials)


def write_labels(tmp_path: Path, rows: list[str], header: str = "case_id,trial,label") -> Path:
    path = tmp_path / "labels.csv"
    path.write_text("# synthetic\n" + header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_read_labels_variants_and_errors(tmp_path: Path) -> None:
    labels = read_labels(write_labels(tmp_path, ["a,0,PASS", "a,1,false", "a,2,1", "a,3,no"]))
    assert [label.passed for label in labels] == [True, False, True, False]
    with pytest.raises(CalibrationError) as info:
        read_labels(write_labels(tmp_path, ["a,0,maybe", "a,x,pass", "a,1,pass", "a,1,fail"]))
    msg = str(info.value)
    assert "line 3: label 'maybe' is not pass/fail" in msg  # true line numbers despite comments
    assert "line 4: trial must be an integer" in msg
    assert "line 6: duplicate label for a trial 1" in msg
    with pytest.raises(CalibrationError, match="missing column"):
        read_labels(write_labels(tmp_path, ["a,0"], header="case_id,trial"))


def test_calibrate_agreement_and_kappa(tmp_path: Path) -> None:
    verdicts = {("a", i): i % 2 == 0 for i in range(10)}  # judge: 5 pass, 5 fail
    run = judged_run(verdicts)
    # humans agree on 8 of 10
    rows = [f"a,{i},{'pass' if (i % 2 == 0) != (i in (1, 2)) else 'fail'}" for i in range(10)]
    rows.append("zzz,0,pass")  # matches nothing
    result = calibrate(run, read_labels(write_labels(tmp_path, rows)))
    (j,) = result.judges
    assert j.agreement.n == 10
    assert j.agreement.observed == pytest.approx(0.8)
    assert j.agreement.kappa == pytest.approx(0.6)
    assert j.agreement.confusion == {"fail": {"fail": 4, "pass": 1}, "pass": {"fail": 1, "pass": 4}}
    assert len(result.unmatched) == 1
    assert "kappa 0.600 (moderate)" in render_calibration_text(result)

    out = tmp_path / "cal.json"
    write_calibration(out, result)
    parsed = CalibrationFile.model_validate_json(out.read_text())
    assert parsed.judges["fp1"].kappa == pytest.approx(0.6)


def test_calibrate_requires_judge_results_and_disambiguation(tmp_path: Path) -> None:
    with pytest.raises(CalibrationError, match="no llm_judge"):
        calibrate(make_run({"a": (1, 1)}), [])
    run = judged_run({("a", 0): True})
    trial = run.trials[0]
    second = AssertionResult("other", "llm_judge", passed=False, score=1.0, fingerprint="fp2")
    doubled = Run(
        meta=run.meta,
        trials=[trial.model_copy(update={"assertions": [*trial.assertions, second]})],
    )
    with pytest.raises(CalibrationError, match="add an `assertion` column"):
        calibrate(doubled, read_labels(write_labels(tmp_path, ["a,0,pass"])))
    labels = read_labels(
        write_labels(tmp_path, ["a,0,pass,other"], header="case_id,trial,label,assertion")
    )
    result = calibrate(doubled, labels)
    assert [j.name for j in result.judges] == ["other"]
