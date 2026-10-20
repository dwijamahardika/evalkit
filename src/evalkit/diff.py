"""Compare two runs and decide what actually changed.

Per case: Fisher's exact test on pass/fail counts (exact at the small n that LLM
evals can afford), Holm-adjusted across cases so a 40-case suite does not
"regress" two cases by chance on every PR.

Suite level: the mean per-case pass-rate difference with a paired bootstrap CI
over cases, which accounts for the fact that cases differ wildly in difficulty
(pooling all trials as if they were one coin would understate the uncertainty).
"""

from __future__ import annotations

import enum
from collections import Counter
from dataclasses import dataclass, field
from typing import Literal

from evalkit.report import EXIT_GATE_FAILED, EXIT_INFRA, EXIT_OK, fmt_pct
from evalkit.runstore import Run, RunMeta
from evalkit.stats import (
    Interval,
    cluster_bootstrap_delta,
    fisher_exact,
    holm_bonferroni,
    smallest_detectable_drop,
    wilson_interval,
)

Status = Literal["regressed", "improved", "unchanged", "added", "removed"]


class FailOn(enum.StrEnum):
    ANY = "any"  # a significant case regression or a suite-level regression
    CASE = "case"
    SUITE = "suite"
    NONE = "none"


@dataclass(frozen=True)
class CaseDiff:
    id: str
    status: Status
    base: tuple[int, int] | None  # (passed, trials)
    cand: tuple[int, int] | None
    p_value: float | None = None
    p_adjusted: float | None = None

    @property
    def delta(self) -> float | None:
        if self.base is None or self.cand is None or not self.base[1] or not self.cand[1]:
            return None
        return self.cand[0] / self.cand[1] - self.base[0] / self.base[1]


@dataclass(frozen=True)
class SuiteDiff:
    base: RunMeta
    cand: RunMeta
    cases: list[CaseDiff]
    alpha: float
    delta: Interval | None  # None when the runs share no cases
    base_overall: Interval
    cand_overall: Interval
    resamples: int
    detectable_drop: float | None
    typical_trials: tuple[int, int] | None
    errors: tuple[int, int]  # errored trials in (base, cand)
    warnings: list[str] = field(default_factory=list)

    @property
    def regressions(self) -> list[CaseDiff]:
        return [c for c in self.cases if c.status == "regressed"]

    @property
    def improvements(self) -> list[CaseDiff]:
        return [c for c in self.cases if c.status == "improved"]

    @property
    def suite_status(self) -> str:
        if self.delta is None:
            return "not comparable"
        if self.delta.high < 0:
            return "regressed"
        if self.delta.low > 0:
            return "improved"
        return "no significant change"

    def exit_code(self, fail_on: FailOn = FailOn.ANY, *, allow_errors: bool = False) -> int:
        if any(self.errors) and not allow_errors:
            return EXIT_INFRA
        case_bad = bool(self.regressions)
        suite_bad = self.suite_status == "regressed"
        failed = {
            FailOn.ANY: case_bad or suite_bad,
            FailOn.CASE: case_bad,
            FailOn.SUITE: suite_bad,
            FailOn.NONE: False,
        }[fail_on]
        return EXIT_GATE_FAILED if failed else EXIT_OK


def _counts(run: Run) -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    for case_id, trials in run.by_case().items():
        out[case_id] = (sum(t.passed for t in trials), len(trials))
    return out


def diff_runs(
    base: Run, cand: Run, *, alpha: float = 0.05, resamples: int = 10_000, seed: int = 0
) -> SuiteDiff:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    a, b = _counts(base), _counts(cand)
    order = [c.id for c in base.meta.cases] + [c.id for c in cand.meta.cases if c.id not in a]
    common = [cid for cid in order if cid in a and cid in b and a[cid][1] and b[cid][1]]

    p_values = []
    for cid in common:
        (ka, na), (kb, nb) = a[cid], b[cid]
        p_values.append(fisher_exact(ka, na - ka, kb, nb - kb))
    adjusted = holm_bonferroni(p_values)
    tested = dict(zip(common, zip(p_values, adjusted, strict=True), strict=True))

    cases: list[CaseDiff] = []
    for cid in order:
        if cid not in b:
            cases.append(CaseDiff(cid, "removed", a.get(cid), None))
        elif cid not in a:
            cases.append(CaseDiff(cid, "added", None, b[cid]))
        elif cid not in tested:
            cases.append(CaseDiff(cid, "unchanged", a[cid], b[cid]))
        else:
            p, p_adj = tested[cid]
            status: Status = "unchanged"
            ra, rb = a[cid][0] / a[cid][1], b[cid][0] / b[cid][1]
            if p_adj < alpha and rb < ra:
                status = "regressed"
            elif p_adj < alpha and rb > ra:
                status = "improved"
            cases.append(CaseDiff(cid, status, a[cid], b[cid], p, p_adj))

    pairs = [(*a[c], *b[c]) for c in common]
    delta = (
        cluster_bootstrap_delta(pairs, resamples=resamples, confidence=1.0 - alpha, seed=seed)
        if pairs
        else None
    )

    n_pairs = Counter((a[c][1], b[c][1]) for c in common)
    typical = n_pairs.most_common(1)[0][0] if n_pairs else None
    detectable = (
        smallest_detectable_drop(typical[0], typical[1], alpha=alpha, comparisons=len(common))
        if typical
        else None
    )

    warnings: list[str] = []
    if base.meta.suite != cand.meta.suite:
        warnings.append(f"different suites: {base.meta.suite!r} vs {cand.meta.suite!r}")
    removed = [c.id for c in cases if c.status == "removed"]
    added = [c.id for c in cases if c.status == "added"]
    if removed:
        warnings.append(f"{len(removed)} case(s) only in base: {', '.join(removed)}")
    if added:
        warnings.append(f"{len(added)} case(s) only in candidate: {', '.join(added)}")
    errors = (
        sum(t.error is not None for t in base.trials),
        sum(t.error is not None for t in cand.trials),
    )
    if any(errors):
        warnings.append(
            f"errored trials (counted as failures): base {errors[0]}, candidate {errors[1]}"
        )

    def pooled(counts: dict[str, tuple[int, int]]) -> Interval:
        return wilson_interval(sum(counts[c][0] for c in common), sum(counts[c][1] for c in common))

    return SuiteDiff(
        base=base.meta,
        cand=cand.meta,
        cases=cases,
        alpha=alpha,
        delta=delta,
        base_overall=pooled(a),
        cand_overall=pooled(b),
        resamples=resamples,
        detectable_drop=detectable,
        typical_trials=typical,
        errors=errors,
        warnings=warnings,
    )


# ---------------------------------------------------------------------- rendering


def _md(text: str) -> str:
    return text.replace("|", "\\|")


def _pp(x: float) -> str:
    return f"{100.0 * x:+.1f} pp"


def _counts_str(c: tuple[int, int] | None) -> str:
    return "-" if c is None else f"{c[0]}/{c[1]}"


def _p(x: float | None) -> str:
    if x is None:
        return "-"
    return f"{x:.2g}" if x >= 0.001 else f"{x:.1e}"


def _run_label(m: RunMeta) -> str:
    return f"{m.provider.get('type')}/{m.provider.get('model')} (run {m.run_id})"


def _summary_line(d: SuiteDiff) -> str:
    head = f"pass rate {fmt_pct(d.base_overall.estimate)} -> {fmt_pct(d.cand_overall.estimate)}"
    if d.delta is None:
        return head
    n = sum(1 for c in d.cases if c.p_value is not None)
    conf = round(100 * (1 - d.alpha))
    return (
        f"{head}; mean per-case delta {_pp(d.delta.estimate)}, {conf}% CI "
        f"[{_pp(d.delta.low)}, {_pp(d.delta.high)}] (two-stage bootstrap over {n} cases): "
        f"{d.suite_status}"
    )


def _power_line(d: SuiteDiff) -> str | None:
    if d.typical_trials is None:
        return None
    na, nb = d.typical_trials
    if d.detectable_drop is None:
        return (
            f"power: with {na} vs {nb} trials per case, no single-case drop can reach "
            "significance after correction; add trials."
        )
    floor = 1.0 - d.detectable_drop
    tested = sum(1 for c in d.cases if c.p_value is not None)
    return (
        f"power: with {na} vs {nb} trials per case, a case at 100% must fall to "
        f"<= {fmt_pct(floor)} to be flagged (alpha {d.alpha:g}, Holm over {tested} cases)."
    )


def _verdict(d: SuiteDiff, fail_on: FailOn, allow_errors: bool) -> str:
    code = d.exit_code(fail_on, allow_errors=allow_errors)
    if code == EXIT_INFRA:
        return "ERROR (errored trials; rerun or pass --allow-errors)"
    if code == EXIT_GATE_FAILED:
        parts = []
        if d.regressions:
            parts.append(f"{len(d.regressions)} case(s) regressed")
        if d.suite_status == "regressed":
            parts.append("suite pass rate regressed")
        return "REGRESSION (" + ", ".join(parts) + ")"
    return "OK (no significant regression)"


def render_diff_text(d: SuiteDiff, fail_on: FailOn = FailOn.ANY, allow_errors: bool = False) -> str:
    lines = [
        f"diff {d.base.suite}: {_run_label(d.base)} -> {_run_label(d.cand)}",
        f"  {_summary_line(d)}",
    ]
    changed = [c for c in d.cases if c.status != "unchanged" or (c.delta or 0.0) != 0.0]
    width = max([len(c.id) for c in changed] + [4])
    for c in changed:
        delta = "" if c.delta is None else _pp(c.delta)
        lines.append(
            f"  {c.status.upper():<10} {c.id:<{width}}  {_counts_str(c.base):>7} -> "
            f"{_counts_str(c.cand):<7} {delta:>10}  p={_p(c.p_value)}  p_holm={_p(c.p_adjusted)}"
        )
    same = len(d.cases) - len(changed)
    if same:
        lines.append(f"  ({same} case(s) with identical pass counts not shown)")
    power = _power_line(d)
    if power:
        lines.append(f"  {power}")
    lines += [f"  warning: {w}" for w in d.warnings]
    lines.append(f"result: {_verdict(d, fail_on, allow_errors)}")
    return "\n".join(lines)


def render_diff_markdown(
    d: SuiteDiff, fail_on: FailOn = FailOn.ANY, allow_errors: bool = False
) -> str:
    out = [
        f"### evalkit diff: `{d.base.suite}` - **{_verdict(d, fail_on, allow_errors)}**",
        "",
        f"Base: {_run_label(d.base)}  ",
        f"Candidate: {_run_label(d.cand)}",
        "",
        _summary_line(d) + ".",
        "",
    ]
    flagged = [c for c in d.cases if c.status in ("regressed", "improved")]
    if flagged:
        out += [
            "| | Case | Base | Candidate | Delta | p (Fisher) | p (Holm) |",
            "|---|---|---|---|---|---|---|",
        ]
        out += [
            f"| {c.status.upper()} | `{_md(c.id)}` | {_counts_str(c.base)} | "
            f"{_counts_str(c.cand)} | {'' if c.delta is None else _pp(c.delta)} | "
            f"{_p(c.p_value)} | {_p(c.p_adjusted)} |"
            for c in flagged
        ]
        out.append("")
    noise = [
        c for c in d.cases if c.status == "unchanged" and c.delta is not None and c.delta != 0.0
    ]
    if noise:
        out += [
            f"<details><summary>{len(noise)} case(s) moved but not significantly</summary>",
            "",
            "| Case | Base | Candidate | Delta | p (Holm) |",
            "|---|---|---|---|---|",
        ]
        for c in noise:
            assert c.delta is not None
            out.append(
                f"| `{_md(c.id)}` | {_counts_str(c.base)} | {_counts_str(c.cand)} | "
                f"{_pp(c.delta)} | {_p(c.p_adjusted)} |"
            )
        out += ["", "</details>", ""]
    power = _power_line(d)
    if power:
        out.append(f"<sub>{power} Case test: Fisher exact, two-sided.</sub>")
    for w in d.warnings:
        out.append(f"\n> warning: {w}")
    return "\n".join(out) + "\n"
