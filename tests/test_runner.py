from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from evalkit.cache import CacheMode, ResponseCache
from evalkit.config import LoadedSuite, load_suite
from evalkit.providers.base import Completion, CompletionRequest, Provider, TokenUsage
from evalkit.providers.factory import build_provider
from evalkit.report import EXIT_GATE_FAILED, EXIT_INFRA, EXIT_OK, summarize
from evalkit.runner import Runner, RunOptions, SelectionError, render_request
from evalkit.runstore import Run, read_run, write_run
from tests.conftest import EXAMPLE_SUITE, no_sleep


def run_suite(
    loaded: LoadedSuite,
    cache_dir: Path,
    options: RunOptions | None = None,
    provider: Provider | None = None,
    judge: Provider | None = None,
    mode: CacheMode = CacheMode.USE,
) -> tuple[Run, Provider]:
    suite = loaded.suite
    provider = provider or build_provider(suite.provider)
    if judge is None and suite.judge is not None:
        judge = build_provider(suite.judge.provider)
    runner = Runner(
        loaded, provider, judge, ResponseCache(cache_dir, mode), options, sleep=no_sleep
    )
    return asyncio.run(runner.run()), provider


def with_model(loaded: LoadedSuite, model: str) -> LoadedSuite:
    s = loaded.suite
    s = s.model_copy(update={"provider": s.provider.model_copy(update={"model": model})})
    return LoadedSuite(suite=s, path=loaded.path, base_dir=loaded.base_dir)


def counts(run: Run) -> dict[str, tuple[int, int]]:
    return {cid: (sum(t.passed for t in ts), len(ts)) for cid, ts in run.by_case().items()}


def test_example_baseline_and_candidate_numbers(tmp_path: Path) -> None:
    """The numbers the README quotes. If the demo data changes, update the README."""
    loaded = load_suite(EXAMPLE_SUITE)
    base, _ = run_suite(loaded, tmp_path)
    assert counts(base) == {
        "minimarket-cash": (9, 10),
        "warung-ribuan": (10, 10),
        "apotek-ppn": (10, 10),
        "toko-bangunan-large": (10, 10),
        "kedai-kopi-diskon": (10, 10),  # the invented item is caught by an *advisory* judge
        "laundry-qris": (10, 10),
        "bengkel-transfer": (8, 10),
        "restoran-service-pb1": (10, 10),
    }
    assert summarize(base).exit_code() == EXIT_OK
    judges = base.meta.judges
    assert len(judges) == 1
    assert judges[0]["gating"] is False
    assert "kappa 0.49 < min_kappa 0.60" in judges[0]["note"]

    cand, _ = run_suite(with_model(loaded, "fake-extractor-b"), tmp_path)
    assert counts(cand)["toko-bangunan-large"] == (0, 10)
    assert sum(p for p, _ in counts(cand).values()) == 67
    assert summarize(cand).exit_code() == EXIT_GATE_FAILED


def test_records_are_ordered_and_complete(tmp_path: Path) -> None:
    run, _ = run_suite(load_suite(EXAMPLE_SUITE), tmp_path, RunOptions(trials=3))
    keys = [(t.case_id, t.trial) for t in run.trials]
    order = [c.id for c in run.meta.cases]
    assert keys == [(c, i) for c in order for i in range(3)]
    first = run.trials[0]
    assert first.output is not None
    assert first.served_model == "fake-extractor-a"
    assert first.cost_usd is not None
    assert first.cost_usd > 0
    assert first.judge_cost_usd is not None
    assert first.judge_cost_usd > 0
    assert {a.type for a in first.assertions} >= {"json_schema", "llm_judge", "numeric_close"}
    # apotek-ppn is scripted to fail twice (429, 503) per trial before succeeding
    apotek = [t for t in run.trials if t.case_id == "apotek-ppn"]
    assert {t.attempts for t in apotek} == {3}
    assert all(t.passed for t in apotek)


def test_cache_makes_reruns_free_and_identical(tmp_path: Path) -> None:
    loaded = load_suite(EXAMPLE_SUITE)
    first, _ = run_suite(loaded, tmp_path, RunOptions(trials=4))
    second, p2 = run_suite(loaded, tmp_path, RunOptions(trials=4))
    assert getattr(p2, "calls") == 0  # noqa: B009 - FakeProvider attribute
    assert second.meta.cache["misses"] == 0
    assert [t.output for t in first.trials] == [t.output for t in second.trials]
    assert all(t.cached for t in second.trials)
    assert all(t.attempts == 0 for t in second.trials)
    # Raising the trial count only pays for the new samples.
    third, _ = run_suite(loaded, tmp_path, RunOptions(trials=5))
    assert all(t.cached == (t.trial < 4) for t in third.trials)
    # refresh ignores the cache
    _, p4 = run_suite(loaded, tmp_path, RunOptions(trials=1), mode=CacheMode.REFRESH)
    assert getattr(p4, "calls") > 0  # noqa: B009


def test_selection(tmp_path: Path) -> None:
    loaded = load_suite(EXAMPLE_SUITE)
    run, _ = run_suite(loaded, tmp_path, RunOptions(trials=1, case_ids=frozenset({"laundry-qris"})))
    assert [c.id for c in run.meta.cases] == ["laundry-qris"]
    run, _ = run_suite(loaded, tmp_path, RunOptions(trials=1, tags=frozenset({"tax"})))
    assert [c.id for c in run.meta.cases] == ["apotek-ppn", "restoran-service-pb1"]
    with pytest.raises(SelectionError, match="unknown case id"):
        run_suite(loaded, tmp_path, RunOptions(case_ids=frozenset({"nope"})))
    with pytest.raises(SelectionError, match="no cases"):
        run_suite(loaded, tmp_path, RunOptions(tags=frozenset({"nope"})))


SIMPLE = """
version: 1
name: simple
provider: {{type: fake, model: m, script: model.yaml}}
{judge}
run: {{trials: 4, concurrency: 2, max_attempts: 3}}
prompt:
  messages: [{{role: user, content: "q {{{{ x }}}}"}}]
defaults:
  assert:
{asserts}
cases:
  - id: c1
    inputs: {{x: one}}
"""


def simple_suite(
    write_suite: Callable[..., Path],
    model_script: str,
    asserts: str = "    - {type: contains, value: ok}",
    judge: str = "",
    files: dict[str, str] | None = None,
) -> LoadedSuite:
    body = SIMPLE.format(judge=judge, asserts=asserts)
    return load_suite(write_suite(body, {"model.yaml": model_script, **(files or {})}))


def test_provider_failure_is_recorded_as_error(
    write_suite: Callable[..., Path], tmp_path: Path
) -> None:
    loaded = simple_suite(
        write_suite,
        "rules:\n  - when: q\n    fail_attempts: [503, 503, 503]\n    responses: [ok]\n",
    )
    run, _ = run_suite(loaded, tmp_path / "cache")
    assert all(t.error and "HTTP 503" in t.error for t in run.trials)
    assert all(t.attempts == 3 and t.output is None for t in run.trials)
    summary = summarize(run)
    assert summary.exit_code() == EXIT_INFRA
    assert summary.exit_code(allow_errors=True) == EXIT_GATE_FAILED


JUDGE_BLOCK = """judge:
  provider: {{type: fake, model: j, script: judge.yaml}}
  {extra}"""


def test_gating_judge_decides_pass_fail(write_suite: Callable[..., Path], tmp_path: Path) -> None:
    loaded = simple_suite(
        write_suite,
        "rules:\n  - responses: [ok]\n",
        asserts="    - {type: llm_judge, rubric: 'is {{ x }} ok?', threshold: 4}",
        judge=JUDGE_BLOCK.format(extra=""),
        files={"judge.yaml": 'default: \'{"reason": "meh", "score": 3}\'\n'},
    )
    run, _ = run_suite(loaded, tmp_path / "cache")
    assert not any(t.passed for t in run.trials)
    result = run.trials[0].assertions[0]
    assert (result.gating, result.score, result.passed) == (True, 3.0, False)
    assert result.detail == "score 3/5 (threshold 4): meh"


def test_unparseable_judge_is_an_infra_error(
    write_suite: Callable[..., Path], tmp_path: Path
) -> None:
    loaded = simple_suite(
        write_suite,
        "rules:\n  - responses: [ok]\n",
        asserts="    - {type: llm_judge, rubric: r}",
        judge=JUDGE_BLOCK.format(extra=""),
        files={"judge.yaml": "default: 'I think it is fine'\n"},
    )
    run, _ = run_suite(loaded, tmp_path / "cache")
    assert all(t.error and "no JSON object" in t.error for t in run.trials)
    assert summarize(run).exit_code() == EXIT_INFRA


class InFlight(Provider):
    name = "inflight"

    def __init__(self) -> None:
        super().__init__("m")
        self.now = 0
        self.peak = 0

    def identity(self) -> dict[str, Any]:
        return {"type": self.name}

    async def complete(self, request: CompletionRequest) -> Completion:
        self.now += 1
        self.peak = max(self.peak, self.now)
        await asyncio.sleep(0.005)
        self.now -= 1
        return Completion("ok", 5.0, (TokenUsage(model="m"),), "m")


def test_concurrency_limit_is_respected(write_suite: Callable[..., Path], tmp_path: Path) -> None:
    loaded = simple_suite(write_suite, "rules: []\n")
    provider = InFlight()
    run, _ = run_suite(
        loaded,
        tmp_path / "c",
        RunOptions(trials=20, concurrency=3),
        provider=provider,
        mode=CacheMode.OFF,
    )
    assert len(run.trials) == 20
    assert provider.peak == 3


def test_render_request_and_run_roundtrip(tmp_path: Path) -> None:
    loaded = load_suite(EXAMPLE_SUITE)
    case = loaded.suite.cases[0]
    request = render_request(loaded, case, 4)
    assert request.sample_index == 4
    assert "Total belanja                Rp 45.500" in request.messages[0].content
    assert "{{" not in request.prompt_text()

    run, _ = run_suite(loaded, tmp_path / "cache", RunOptions(trials=2))
    path = tmp_path / "out" / "run.jsonl"
    write_run(path, run)
    again = read_run(path)
    assert again.meta == run.meta
    assert again.trials == run.trials
