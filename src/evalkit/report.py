"""Summaries of one run: terminal text, Markdown (PR comment) and JUnit XML."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field

from evalkit.runstore import Run, TrialRecord
from evalkit.stats import Interval, percentile, wilson_interval

EXIT_OK = 0
EXIT_GATE_FAILED = 1
EXIT_USAGE = 2
EXIT_INFRA = 3

_EPS = 1e-12


def fmt_pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def fmt_ci(ci: Interval) -> str:
    return f"{100.0 * ci.low:.1f}-{100.0 * ci.high:.1f}%"


def fmt_usd(x: float | None) -> str:
    return "n/a" if x is None else f"${x:.4f}"


def _sum_or_none(values: list[float | None]) -> float | None:
    if any(v is None for v in values):
        return None
    return sum(v for v in values if v is not None)


@dataclass(frozen=True)
class CaseSummary:
    id: str
    trials: int
    passed: int
    errors: int
    rate: Interval
    min_pass_rate: float
    failures: dict[str, int]  # gating assertion name -> failed trials
    advisory_failures: dict[str, int]
    example: TrialRecord | None  # first failing trial
    latency_p50: float | None
    latency_p95: float | None
    cost_usd: float | None
    judge_cost_usd: float | None
    refusals: int
    fallback_served: int

    @property
    def gate_passed(self) -> bool:
        return self.trials > 0 and self.rate.estimate + _EPS >= self.min_pass_rate


@dataclass(frozen=True)
class RunSummary:
    run: Run
    cases: list[CaseSummary]
    overall: Interval
    trials: int
    passed: int
    errors: int
    cost_usd: float | None
    judge_cost_usd: float | None
    latency_p50: float | None
    latency_p95: float | None
    advisory_judges: list[dict[str, object]] = field(default_factory=list)

    @property
    def suite_gate_passed(self) -> bool:
        floor = self.run.meta.min_suite_pass_rate
        return floor is None or self.overall.estimate + _EPS >= floor

    @property
    def gate_passed(self) -> bool:
        return all(c.gate_passed for c in self.cases) and self.suite_gate_passed

    def exit_code(self, *, allow_errors: bool = False) -> int:
        if self.errors and not allow_errors:
            return EXIT_INFRA
        return EXIT_OK if self.gate_passed else EXIT_GATE_FAILED

    def verdict(self, *, allow_errors: bool = False) -> str:
        code = self.exit_code(allow_errors=allow_errors)
        return {EXIT_OK: "PASS", EXIT_GATE_FAILED: "FAIL", EXIT_INFRA: "ERROR"}[code]


def _latencies(trials: list[TrialRecord]) -> list[float]:
    return [t.latency_ms for t in trials if t.latency_ms is not None]


def summarize(run: Run) -> RunSummary:
    model = run.meta.provider.get("model")
    grouped = run.by_case()
    cases: list[CaseSummary] = []
    for meta in run.meta.cases:
        trials = sorted(grouped.get(meta.id, []), key=lambda t: t.trial)
        passed = sum(t.passed for t in trials)
        failures: Counter[str] = Counter()
        advisory: Counter[str] = Counter()
        for t in trials:
            for a in t.assertions:
                if not a.passed:
                    (failures if a.gating else advisory)[a.name] += 1
        lat = _latencies(trials)
        cases.append(
            CaseSummary(
                id=meta.id,
                trials=len(trials),
                passed=passed,
                errors=sum(t.error is not None for t in trials),
                rate=wilson_interval(passed, len(trials)),
                min_pass_rate=meta.min_pass_rate,
                failures=dict(failures.most_common()),
                advisory_failures=dict(advisory.most_common()),
                example=next((t for t in trials if not t.passed), None),
                latency_p50=percentile(lat, 0.5) if lat else None,
                latency_p95=percentile(lat, 0.95) if lat else None,
                # Unknown (None) if any priced call lacks a price; trials whose model
                # call failed have no completion and contribute nothing.
                cost_usd=_sum_or_none([t.cost_usd for t in trials if t.output is not None]),
                judge_cost_usd=_sum_or_none(
                    [
                        t.judge_cost_usd
                        for t in trials
                        if any(a.type == "llm_judge" for a in t.assertions)
                    ]
                ),
                refusals=sum(t.stop_reason == "refusal" for t in trials),
                fallback_served=sum(
                    t.served_model is not None and t.served_model != model for t in trials
                ),
            )
        )
    total = sum(c.trials for c in cases)
    passed = sum(c.passed for c in cases)
    lat = _latencies(run.trials)
    return RunSummary(
        run=run,
        cases=cases,
        overall=wilson_interval(passed, total),
        trials=total,
        passed=passed,
        errors=sum(c.errors for c in cases),
        cost_usd=_sum_or_none([c.cost_usd for c in cases]),
        judge_cost_usd=_sum_or_none([c.judge_cost_usd for c in cases]),
        latency_p50=percentile(lat, 0.5) if lat else None,
        latency_p95=percentile(lat, 0.95) if lat else None,
        advisory_judges=[j for j in run.meta.judges if not j.get("gating", True)],
    )


def _status(case: CaseSummary) -> str:
    if case.errors:
        return "ERROR"
    return "PASS" if case.gate_passed else "FAIL"


def _top_failure(case: CaseSummary) -> str:
    if case.errors:
        return f"{case.errors} trial(s) errored"
    failures, prefix = case.failures, ""
    if not failures:
        failures, prefix = case.advisory_failures, "advisory: "
    if not failures:
        return ""
    name, count = next(iter(failures.items()))
    more = f" (+{len(failures) - 1} more)" if len(failures) > 1 else ""
    return f"{prefix}{name} x{count}{more}"


def _header(summary: RunSummary) -> str:
    m = summary.run.meta
    trials = sorted({c.trials for c in m.cases})
    per_case = str(trials[0]) if len(trials) == 1 else f"{trials[0]}-{trials[-1]}"
    return (
        f"{m.suite} | {m.provider.get('type')}/{m.provider.get('model')} | "
        f"{len(m.cases)} cases x {per_case} trials | run {m.run_id}"
    )


def _notes(summary: RunSummary) -> list[str]:
    notes = [
        f"judge `{j.get('name')}` is advisory only ({j.get('note')})"
        for j in summary.advisory_judges
    ]
    refusals = sum(c.refusals for c in summary.cases)
    if refusals:
        notes.append(f"{refusals} trial(s) ended with stop_reason=refusal")
    served = sum(c.fallback_served for c in summary.cases)
    if served:
        notes.append(f"{served} trial(s) were served by a different model (server-side fallback)")
    floor = summary.run.meta.min_suite_pass_rate
    if floor is not None and not summary.suite_gate_passed:
        notes.append(f"overall pass rate {fmt_pct(summary.overall.estimate)} < {fmt_pct(floor)}")
    return notes


def render_text(summary: RunSummary, *, allow_errors: bool = False) -> str:
    width = max([len(c.id) for c in summary.cases] + [4])
    lines = [_header(summary)]
    lines += [
        f"  {_status(c):<5} {c.id:<{width}}  {c.passed:>3}/{c.trials:<3} "
        f"{fmt_pct(c.rate.estimate):>6}  {'[' + fmt_ci(c.rate) + ']':<14} "
        f"gate>={fmt_pct(c.min_pass_rate):<6}  {_top_failure(c)}".rstrip()
        for c in summary.cases
    ]
    lat = (
        f" | p50 {summary.latency_p50:.0f} ms, p95 {summary.latency_p95:.0f} ms"
        if summary.latency_p50 is not None and summary.latency_p95 is not None
        else ""
    )
    cache = summary.run.meta.cache
    judge = f" + judge {fmt_usd(summary.judge_cost_usd)}" if summary.run.meta.judge else ""
    lines.append(
        f"overall {summary.passed}/{summary.trials} ({fmt_pct(summary.overall.estimate)}, "
        f"95% CI {fmt_ci(summary.overall)}) | cost {fmt_usd(summary.cost_usd)}{judge}{lat} | "
        f"cache {cache.get('hits', 0)} hit / {cache.get('misses', 0)} miss"
    )
    lines += [f"note: {n}" for n in _notes(summary)]
    lines.append(f"result: {summary.verdict(allow_errors=allow_errors)}")
    return "\n".join(lines)


def _md_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _clip(text: str | None, limit: int = 600) -> str:
    if text is None:
        return ""
    return text if len(text) <= limit else text[:limit] + "\n... (truncated)"


def render_markdown(summary: RunSummary, *, allow_errors: bool = False) -> str:
    m = summary.run.meta
    out = [
        f"### evalkit: `{m.suite}` - **{summary.verdict(allow_errors=allow_errors)}**",
        "",
        f"{_header(summary)}",
        "",
        "| | Case | Pass rate | 95% CI | Gate | Top failure |",
        "|---|---|---|---|---|---|",
    ]
    out += [
        f"| {_status(c)} | `{_md_escape(c.id)}` | {c.passed}/{c.trials} "
        f"({fmt_pct(c.rate.estimate)}) | "
        f"{fmt_ci(c.rate)} | >= {fmt_pct(c.min_pass_rate)} | {_md_escape(_top_failure(c))} |"
        for c in summary.cases
    ]
    out += [
        "",
        f"**Overall:** {summary.passed}/{summary.trials} trials passed "
        f"({fmt_pct(summary.overall.estimate)}, 95% CI {fmt_ci(summary.overall)}, "
        "pooled over trials).",
        f"**Cost:** model {fmt_usd(summary.cost_usd)}"
        + (f", judge {fmt_usd(summary.judge_cost_usd)}" if m.judge else "")
        + " (from the suite's price table; verify prices before relying on them).",
    ]
    if summary.latency_p50 is not None and summary.latency_p95 is not None:
        out.append(
            f"**Latency:** p50 {summary.latency_p50:.0f} ms, p95 {summary.latency_p95:.0f} ms."
        )
    notes = _notes(summary)
    if notes:
        out += ["", *[f"> {n}" for n in notes]]
    failing = [c for c in summary.cases if c.example is not None]
    if failing:
        out += ["", "<details><summary>Failure details</summary>", ""]
        for c in failing:
            ex = c.example
            assert ex is not None
            out.append(f"**`{_md_escape(c.id)}`** - trial {ex.trial}")
            if ex.error:
                out.append(f"- error: {_md_escape(ex.error)}")
            for a in ex.assertions:
                if not a.passed:
                    tag = "" if a.gating else " (advisory)"
                    out.append(f"- `{_md_escape(a.name)}`{tag}: {_md_escape(a.detail)}")
            if ex.output is not None:
                out += ["", "```text", _clip(ex.output).replace("```", "'''"), "```", ""]
        out.append("</details>")
    return "\n".join(out) + "\n"


def render_junit(summary: RunSummary) -> str:
    m = summary.run.meta
    failures = sum(1 for c in summary.cases if not c.errors and not c.gate_passed)
    errors = sum(1 for c in summary.cases if c.errors)
    total_time = sum((t.latency_ms or 0.0) for t in summary.run.trials) / 1000.0
    root = ET.Element(
        "testsuites",
        name="evalkit",
        tests=str(len(summary.cases)),
        failures=str(failures),
        errors=str(errors),
        time=f"{total_time:.3f}",
    )
    suite_el = ET.SubElement(
        root,
        "testsuite",
        name=m.suite,
        tests=str(len(summary.cases)),
        failures=str(failures),
        errors=str(errors),
        skipped="0",
        time=f"{total_time:.3f}",
        timestamp=m.started_at,
    )
    props = ET.SubElement(suite_el, "properties")
    for key, value in (
        ("run_id", m.run_id),
        ("provider", str(m.provider.get("type"))),
        ("model", str(m.provider.get("model"))),
        ("overall_pass_rate", f"{summary.overall.estimate:.4f}"),
    ):
        ET.SubElement(props, "property", name=key, value=value)
    by_case = summary.run.by_case()
    for c in summary.cases:
        case_time = sum((t.latency_ms or 0.0) for t in by_case.get(c.id, [])) / 1000.0
        tc = ET.SubElement(
            suite_el, "testcase", classname=m.suite, name=c.id, time=f"{case_time:.3f}"
        )
        stats = (
            f"pass rate {c.passed}/{c.trials} ({fmt_pct(c.rate.estimate)}), "
            f"95% CI {fmt_ci(c.rate)}, gate >= {fmt_pct(c.min_pass_rate)}"
        )
        detail_lines = [f"{name}: failed in {n} trial(s)" for name, n in c.failures.items()]
        if c.example is not None:
            detail_lines += ["", f"first failing trial ({c.example.trial}):"]
            detail_lines += [
                f"  {a.name}: {a.detail}" for a in c.example.assertions if not a.passed
            ]
            if c.example.error:
                detail_lines.append(f"  error: {c.example.error}")
        if c.errors:
            el = ET.SubElement(
                tc, "error", message=f"{c.errors} trial(s) errored", type="ProviderError"
            )
            el.text = "\n".join(detail_lines)
        elif not c.gate_passed:
            el = ET.SubElement(tc, "failure", message=stats, type="PassRateBelowGate")
            el.text = "\n".join(detail_lines)
        ET.SubElement(tc, "system-out").text = stats
    ET.indent(root)
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding="unicode") + "\n"
