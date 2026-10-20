"""End-to-end through the CLI, offline, on the example suite."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path

import pytest
from typer.testing import CliRunner

from evalkit import __version__
from evalkit.cli import app
from evalkit.config import LlmJudge, load_suite
from evalkit.judge import fingerprint
from evalkit.providers.factory import build_provider
from tests.conftest import EXAMPLE_LABELS, EXAMPLE_SUITE, MINIMAL_SCRIPT

runner = CliRunner()


def invoke(*args: str | Path) -> tuple[int, str]:
    result = runner.invoke(app, [str(a) for a in args])
    return result.exit_code, result.output


@pytest.fixture
def cache(tmp_path: Path) -> list[str]:
    return ["--cache-dir", str(tmp_path / "cache")]


def test_version() -> None:
    assert invoke("--version") == (0, f"evalkit {__version__}\n")


def test_validate(write_suite: Callable[..., Path]) -> None:
    code, out = invoke("validate", EXAMPLE_SUITE)
    assert code == 0
    assert "ok (8 cases, 57 assertions, fake/fake-extractor-a, 10 trials/case)" in out
    bad = write_suite("name: x\nprovider: {type: nope}\nprompt: {messages: []}\ncases: []\n")
    code, out = invoke("validate", EXAMPLE_SUITE, bad)
    assert code == 2
    assert "errors" in out
    assert "provider" in out
    assert "cases" in out


def test_demo_flow(tmp_path: Path, cache: list[str]) -> None:
    base, cand = tmp_path / "base.jsonl", tmp_path / "cand.jsonl"
    code, out = invoke("run", EXAMPLE_SUITE, "--out", base, "--label", "variant=baseline", *cache)
    assert code == 0, out
    assert "result: PASS" in out
    assert "judge `faithful_items` is advisory only" in out

    md, junit = tmp_path / "cand.md", tmp_path / "cand.xml"
    code, out = invoke(
        "run",
        EXAMPLE_SUITE,
        "--model",
        "fake-extractor-b",
        "--out",
        cand,
        "--markdown",
        md,
        "--junit",
        junit,
        "-q",
        *cache,
    )
    assert code == 1, out
    assert "FAIL  toko-bangunan-large" in out
    assert "numeric_close(total~1250000) x10" in out
    assert "### evalkit: `receipts-extraction` - **FAIL**" in md.read_text()
    suite_el = ET.parse(junit).getroot().find("testsuite")
    assert suite_el is not None
    assert suite_el.get("failures") == "1"

    header = json.loads(base.read_text().splitlines()[0])
    assert header["labels"] == {"variant": "baseline"}

    diff_md = tmp_path / "diff.md"
    code, out = invoke("diff", base, cand, "--markdown", diff_md)
    assert code == 1, out
    assert "REGRESSED  toko-bangunan-large" in out
    assert "result: REGRESSION (1 case(s) regressed)" in out
    assert "| REGRESSED | `toko-bangunan-large` |" in diff_md.read_text()
    assert invoke("diff", base, cand, "--fail-on", "none")[0] == 0
    assert invoke("diff", base, base)[0] == 0

    assert invoke("report", base)[0] == 0
    code, out = invoke("report", cand)
    assert code == 1
    assert "overall pass rate 83.8% < 90.0%" in out

    cal = tmp_path / "cal.json"
    code, out = invoke("calibrate", base, EXAMPLE_LABELS, "--out", cal, "--min-kappa", "0.6")
    assert code == 1, out
    assert "agreement 97.5%  kappa 0.487 (moderate)" in out
    assert json.loads(cal.read_text())["judges"]
    assert invoke("calibrate", base, EXAMPLE_LABELS, "--min-kappa", "0.4")[0] == 0


def test_committed_calibration_matches_the_suite() -> None:
    """Editing the rubric or judge without re-running `calibrate` must not go unnoticed."""
    loaded = load_suite(EXAMPLE_SUITE)
    judge_cfg = loaded.suite.judge
    assert judge_cfg is not None
    assert judge_cfg.calibration is not None
    data = json.loads((loaded.base_dir / judge_cfg.calibration).read_text())
    rubric = next(a for a in loaded.suite.defaults.assert_ if a.type == "llm_judge")
    assert isinstance(rubric, LlmJudge)
    fp = fingerprint(build_provider(judge_cfg.provider).identity(), rubric)
    assert fp in data["judges"]
    assert data["judges"][fp]["kappa"] == pytest.approx(0.487, abs=0.001)


def test_run_input_errors(
    tmp_path: Path, cache: list[str], write_suite: Callable[..., Path]
) -> None:
    out = tmp_path / "r.jsonl"
    code, text = invoke("run", EXAMPLE_SUITE, "--out", out, "--label", "novalue", *cache)
    assert (code, "--label expects key=value" in text) == (2, True)
    code, text = invoke("run", EXAMPLE_SUITE, "--out", out, "--case", "nope", *cache)
    assert (code, "unknown case id(s): nope" in text) == (2, True)
    code, text = invoke("run", EXAMPLE_SUITE, "--out", out, "--model", "unpriced", *cache)
    assert (code, "no price for model 'unpriced'" in text) == (2, True)

    openai_suite = write_suite(
        """
        name: o
        provider:
          type: openai
          model: m
          base_url: "http://127.0.0.1:9/v1"
          api_key_env: EVALKIT_TEST_UNSET
        prompt: {messages: [{role: user, content: hi}]}
        cases: [{id: a, assert: [{type: contains, value: x}]}]
        """
    )
    code, text = invoke("run", openai_suite, "--out", out, *cache)
    assert (code, "EVALKIT_TEST_UNSET is not set" in text) == (2, True)

    garbage = tmp_path / "garbage.jsonl"
    garbage.write_text('{"type": "trial"}\n')
    assert invoke("report", garbage)[0] == 2
    assert invoke("diff", garbage, garbage)[0] == 2
    assert invoke("report", tmp_path / "missing.jsonl")[0] == 2


def test_infra_errors_exit_3(
    tmp_path: Path, cache: list[str], write_suite: Callable[..., Path]
) -> None:
    suite = write_suite(
        """
        name: flaky
        provider: {type: fake, model: m, script: script.yaml}
        run: {trials: 2, max_attempts: 2, backoff_base_s: 0, backoff_cap_s: 0}
        prompt: {messages: [{role: user, content: hello}]}
        cases: [{id: a, assert: [{type: contains, value: HELLO}]}]
        """,
        {"script.yaml": "rules:\n  - fail_attempts: [500, 500]\n    responses: [HELLO]\n"},
    )
    out = tmp_path / "r.jsonl"
    code, text = invoke("run", suite, "--out", out, *cache)
    assert code == 3, text
    assert "result: ERROR" in text
    assert invoke("report", out, "--allow-errors")[0] == 1
    assert invoke("run", suite, "--out", out, "--allow-errors", "--cache", "off", *cache)[0] == 1


def test_minimal_suite_passes(
    tmp_path: Path, cache: list[str], write_suite: Callable[..., Path]
) -> None:
    suite = write_suite(
        """
        name: tiny
        provider: {type: fake, model: m, script: script.yaml}
        prompt: {messages: [{role: user, content: hello}]}
        cases: [{id: a, assert: [{type: equals, value: HELLO}]}]
        """,
        {"script.yaml": MINIMAL_SCRIPT},
    )
    code, text = invoke("run", suite, "--out", tmp_path / "r.jsonl", *cache)
    assert code == 0, text
    assert "PASS  a" in text
