"""Measure evalkit's own overhead and its statistics, offline.

    python -m bench.bench_harness

Uses the fake provider (no network), so the numbers are the harness: prompt
rendering, cache I/O, assertions, JSONL, statistics. Model latency is excluded,
except in the concurrency section, which uses a provider that sleeps.
"""

from __future__ import annotations

import asyncio
import json
import os
import platform
import tempfile
import time
from pathlib import Path
from typing import Any

from evalkit.cache import CacheMode, ResponseCache
from evalkit.config import LoadedSuite, load_suite
from evalkit.diff import diff_runs
from evalkit.providers.base import Completion, CompletionRequest, Provider, TokenUsage
from evalkit.providers.factory import build_provider
from evalkit.runner import Runner, RunOptions
from evalkit.runstore import Run, write_run
from evalkit.stats import fisher_exact

ANSWER = json.dumps(
    {
        "merchant": "Toko Contoh",
        "date": "2026-01-02",
        "currency": "IDR",
        "total": 45500,
        "payment_method": "cash",
        "items": [{"name": "Roti", "qty": 1, "line_total": 45500}],
    }
)


def make_suite(root: Path, cases: int, trials: int, judge: bool) -> LoadedSuite:
    (root / "model.yaml").write_text(json.dumps({"rules": [], "default": ANSWER}))
    (root / "judge.yaml").write_text(json.dumps({"default": '{"reason": "ok", "score": 5}'}))
    schema = {
        "type": "object",
        "required": ["merchant", "total", "items"],
        "properties": {"total": {"type": "integer"}, "items": {"type": "array"}},
    }
    suite: dict[str, Any] = {
        "version": 1,
        "name": "bench",
        "provider": {"type": "fake", "model": "fake", "script": "model.yaml"},
        "run": {"trials": trials, "concurrency": 32},
        "prompt": {
            "system": "Extract the receipt as JSON.",
            "messages": [{"role": "user", "content": "<receipt>{{ receipt }}</receipt>"}],
        },
        "defaults": {
            "assert": [
                {"type": "json_schema", "schema": schema},
                {"type": "numeric_close", "path": "total", "expected": 45500},
                {"type": "equals", "path": "payment_method", "value": "cash"},
                {"type": "regex", "pattern": r"\d{4}-\d{2}-\d{2}"},
                {"type": "max_latency_ms", "value": 1000},
            ]
        },
        "cases": [
            {"id": f"case-{i:04d}", "inputs": {"receipt": f"TOKO CONTOH #{i}\nTOTAL Rp 45.500"}}
            for i in range(cases)
        ],
    }
    if judge:
        suite["judge"] = {"provider": {"type": "fake", "model": "j", "script": "judge.yaml"}}
        suite["defaults"]["assert"].append({"type": "llm_judge", "rubric": "faithful?"})
    path = root / "suite.yaml"
    path.write_text(json.dumps(suite))
    return load_suite(path)


def run_once(
    loaded: LoadedSuite,
    cache_dir: Path,
    mode: CacheMode,
    provider: Provider | None = None,
    concurrency: int | None = None,
) -> tuple[Run, float]:
    s = loaded.suite
    judge = build_provider(s.judge.provider) if s.judge else None
    runner = Runner(
        loaded,
        provider or build_provider(s.provider),
        judge,
        ResponseCache(cache_dir, mode),
        RunOptions(concurrency=concurrency),
    )
    start = time.perf_counter()
    run = asyncio.run(runner.run())
    return run, time.perf_counter() - start


def timed(loaded: LoadedSuite, cache_dir: Path, mode: CacheMode, repeat: int = 3) -> float:
    """Best wall time of `repeat` runs. A fresh cache directory per run when cold."""
    best = float("inf")
    for i in range(repeat):
        target = (
            cache_dir / str(i) if mode is CacheMode.USE and cache_dir.name == "cold" else cache_dir
        )
        best = min(best, run_once(loaded, target, mode)[1])
    return best


class Sleepy(Provider):
    name = "sleepy"

    def __init__(self, latency_s: float) -> None:
        super().__init__("sleepy")
        self.latency_s = latency_s

    def identity(self) -> dict[str, Any]:
        return {"type": self.name}

    async def complete(self, request: CompletionRequest) -> Completion:
        await asyncio.sleep(self.latency_s)
        return Completion(ANSWER, self.latency_s * 1000, (TokenUsage(model="sleepy"),), "sleepy")


def main() -> None:
    cases, trials = 200, 10
    n = cases * trials
    print(
        f"python {platform.python_version()} | {platform.system()} {platform.machine()} | "
        f"{os.cpu_count()} logical CPUs | fake provider"
    )
    print()
    print(f"## Harness overhead: {cases} cases x {trials} trials = {n} trials, 5 assertions each")
    print()
    print("| scenario | wall time | trials/s | us/trial |")
    print("|---|---|---|---|")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for judge in (False, True):
            sub = root / ("judge" if judge else "plain")
            sub.mkdir()
            loaded = make_suite(sub, cases, trials, judge)
            label = " + llm_judge" if judge else ""

            t_off = timed(loaded, sub / "off", CacheMode.OFF)
            t_cold = timed(loaded, sub / "cold", CacheMode.USE)
            run_once(loaded, sub / "warm", CacheMode.USE)  # populate
            t_warm = timed(loaded, sub / "warm", CacheMode.USE)
            for name, t in (
                (f"no cache{label}", t_off),
                (f"cold cache (writes){label}", t_cold),
                (f"warm cache (all hits){label}", t_warm),
            ):
                print(f"| {name} | {t:.2f} s | {n / t:,.0f} | {1e6 * t / n:,.0f} |")

        loaded = make_suite(root / "plain", cases, trials, judge=False)
        run, _ = run_once(loaded, root / "plain" / "warm", CacheMode.USE)
        out = root / "run.jsonl"
        start = time.perf_counter()
        write_run(out, run)
        t_write = time.perf_counter() - start
        print()
        size_kib = out.stat().st_size / 1024
        print(f"Run file: {size_kib:,.0f} KiB for {n} trials, written in {t_write * 1000:.0f} ms.")

        print()
        print("## Concurrency (provider sleeps 20 ms per call, 200 calls, cache off)")
        print()
        print("| concurrency | wall time | speed-up |")
        print("|---|---|---|")
        (root / "conc").mkdir()
        small = make_suite(root / "conc", cases=20, trials=10, judge=False)
        base_t = None
        for c in (1, 8, 32):
            _, t = run_once(
                small, root / "conc" / "c", CacheMode.OFF, provider=Sleepy(0.02), concurrency=c
            )
            base_t = base_t or t
            print(f"| {c} | {t:.2f} s | {base_t / t:.1f}x |")

        print()
        print("## Statistics")
        print()
        print("| operation | time |")
        print("|---|---|")
        for m in (8, 200):
            base = make_counts_run(run, m, drop=False)
            cand = make_counts_run(run, m, drop=True)
            times = []
            for _ in range(3):
                start = time.perf_counter()
                diff_runs(base, cand, resamples=10_000)
                times.append(time.perf_counter() - start)
            print(
                f"| diff, {m} cases x 10 trials, 10k bootstrap resamples "
                f"| {min(times) * 1000:,.0f} ms |"
            )
        start = time.perf_counter()
        for _ in range(100):
            fisher_exact(1000, 0, 950, 50)
        print(
            f"| Fisher exact, 1000 vs 1000 trials | {(time.perf_counter() - start) * 10:.2f} ms |"
        )


def make_counts_run(template: Run, cases: int, drop: bool) -> Run:
    """A run with `cases` cases (10 trials each) derived from a real run's records."""
    by_case = template.by_case()
    ids = list(by_case)[:cases]
    trials = []
    for i, cid in enumerate(ids):
        for t in by_case[cid]:
            fail = drop and i % 7 == 0 and t.trial < 4
            trials.append(t.model_copy(update={"passed": not fail}))
    meta = template.meta.model_copy(
        update={"cases": [c for c in template.meta.cases if c.id in ids]}
    )
    return Run(meta=meta, trials=trials)


if __name__ == "__main__":
    main()
