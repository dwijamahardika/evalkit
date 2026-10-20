"""Statistics used to decide whether a change is a regression or noise.

Everything here is dependency-free and deterministic. The test suite checks the
implementations against scipy as an oracle (property-based, via hypothesis).
"""

from __future__ import annotations

import bisect
import math
import random
from collections import Counter
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from statistics import NormalDist
from typing import Literal

Alternative = Literal["two-sided", "less", "greater"]


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high


def _z(confidence: float) -> float:
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    return NormalDist().inv_cdf(0.5 + confidence / 2.0)


def wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> Interval:
    """Wilson score interval for a binomial proportion.

    Unlike the normal-approximation ("Wald") interval it never leaves [0, 1] and
    stays sensible at 0/n and n/n, which is exactly where LLM pass rates live.
    """
    if trials < 0 or not 0 <= successes <= max(trials, 0):
        raise ValueError(f"invalid counts: {successes}/{trials}")
    if trials == 0:
        return Interval(0.0, 0.0, 1.0)
    z = _z(confidence)
    n = float(trials)
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denom
    low = 0.0 if successes == 0 else max(0.0, center - half)
    high = 1.0 if successes == trials else min(1.0, center + half)
    return Interval(p, low, high)


def fisher_exact(a: int, b: int, c: int, d: int, alternative: Alternative = "two-sided") -> float:
    """Fisher's exact test p-value for the 2x2 table [[a, b], [c, d]].

    Rows are the two runs, columns are (pass, fail). `less`/`greater` refer to the
    odds ratio (a*d)/(b*c), matching scipy.stats.fisher_exact.

    Computed with exact integer binomial coefficients, so ties in the two-sided
    test are detected exactly instead of with a floating-point tolerance.
    """
    if min(a, b, c, d) < 0:
        raise ValueError("table entries must be non-negative")
    row1, row2, col1 = a + b, c + d, a + c
    total = row1 + row2
    if row1 == 0 or row2 == 0 or col1 == 0 or col1 == total:
        return 1.0
    lo, hi = max(0, col1 - row2), min(row1, col1)
    # Numerators of the hypergeometric pmf; the common denominator is C(total, col1).
    weights = {x: math.comb(row1, x) * math.comb(row2, col1 - x) for x in range(lo, hi + 1)}
    denom = math.comb(total, col1)
    observed = weights[a]
    if alternative == "less":
        num = sum(w for x, w in weights.items() if x <= a)
    elif alternative == "greater":
        num = sum(w for x, w in weights.items() if x >= a)
    elif alternative == "two-sided":
        num = sum(w for w in weights.values() if w <= observed)
    else:  # pragma: no cover - guarded by the Literal type
        raise ValueError(f"unknown alternative {alternative!r}")
    return min(1.0, num / denom)


def holm_bonferroni(p_values: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values (controls the family-wise error rate).

    A CI gate that tests every case independently at alpha=0.05 would flag a
    "regression" on roughly 1 in 20 unchanged cases; this keeps the chance of any
    false alarm across the whole suite at alpha.
    """
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[idx]))
        adjusted[idx] = running
    return adjusted


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile (numpy's default), `q` in [0, 1]."""
    if not values:
        raise ValueError("percentile of empty sequence")
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be in [0, 1], got {q}")
    data = sorted(values)
    pos = q * (len(data) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return data[lo]
    frac = pos - lo
    return data[lo] + (data[hi] - data[lo]) * frac


def _binomial_sampler(n: int, p: float) -> Callable[[random.Random], int]:
    """Draw k ~ Binomial(n, p) by inverse CDF (bisect). Built once per case."""
    if n <= 0 or p <= 0.0:
        return lambda rng: 0
    if p >= 1.0:
        return lambda rng: n
    log_p, log_q = math.log(p), math.log1p(-p)
    cdf: list[float] = []
    acc = 0.0
    for k in range(n + 1):
        log_pmf = (
            math.lgamma(n + 1)
            - math.lgamma(k + 1)
            - math.lgamma(n - k + 1)
            + k * log_p
            + (n - k) * log_q
        )
        acc += math.exp(log_pmf)
        cdf.append(acc)
    cdf[-1] = 1.0
    return lambda rng: min(n, bisect.bisect_right(cdf, rng.random()))


def cluster_bootstrap_delta(
    pairs: Sequence[tuple[int, int, int, int]],
    *,
    resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Interval:
    """CI for the mean per-case pass-rate change, from (k_base, n_base, k_cand, n_cand).

    Two-stage bootstrap: resample cases (cases differ wildly in difficulty), then
    resample trials within each chosen case (each pass rate is itself a noisy
    estimate). Resampling cases alone gives a zero-width interval whenever the
    per-case deltas happen to be equal - e.g. a one-case suite going 5/5 -> 4/5
    would be "significant". Two-stage is slightly conservative by construction.
    """
    if not pairs:
        raise ValueError("no paired cases")
    if resamples < 1:
        raise ValueError("resamples must be >= 1")
    for ka, na, kb, nb in pairs:
        if na < 1 or nb < 1 or not 0 <= ka <= na or not 0 <= kb <= nb:
            raise ValueError(f"invalid counts {(ka, na, kb, nb)}")
    m = len(pairs)
    point = sum(kb / nb - ka / na for ka, na, kb, nb in pairs) / m
    samplers = [
        (_binomial_sampler(na, ka / na), na, _binomial_sampler(nb, kb / nb), nb)
        for ka, na, kb, nb in pairs
    ]
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(resamples):
        total = 0.0
        for _ in range(m):
            draw_a, na, draw_b, nb = samplers[rng.randrange(m)]
            total += draw_b(rng) / nb - draw_a(rng) / na
        means.append(total / m)
    tail = (1.0 - confidence) / 2.0
    return Interval(point, percentile(means, tail), percentile(means, 1.0 - tail))


def smallest_detectable_drop(
    n_base: int, n_cand: int, *, alpha: float = 0.05, comparisons: int = 1
) -> float | None:
    """Smallest pass-rate drop from a perfect baseline that the case-level test can flag.

    Answers "with this many trials, could the gate even see a regression?" Returns
    None when no drop (not even to 0%) reaches significance.
    """
    if n_base < 1 or n_cand < 1:
        return None
    threshold = alpha / max(1, comparisons)
    for k in range(n_cand, -1, -1):
        p = fisher_exact(n_base, 0, k, n_cand - k)
        if p < threshold:
            return 1.0 - k / n_cand
    return None


@dataclass(frozen=True)
class Agreement:
    n: int
    observed: float
    kappa: float | None  # None when undefined (both raters used a single label)
    confusion: dict[str, dict[str, int]]  # confusion[rater_a_label][rater_b_label]


def cohen_kappa(pairs: Sequence[tuple[Hashable, Hashable]]) -> Agreement:
    """Cohen's kappa for two raters over the same items."""
    n = len(pairs)
    if n == 0:
        raise ValueError("no rated items")
    labels = sorted({str(x) for pair in pairs for x in pair})
    a_counts = Counter(str(a) for a, _ in pairs)
    b_counts = Counter(str(b) for _, b in pairs)
    confusion = {la: dict.fromkeys(labels, 0) for la in labels}
    for a, b in pairs:
        confusion[str(a)][str(b)] += 1
    observed = sum(confusion[label][label] for label in labels) / n
    expected = sum((a_counts[label] / n) * (b_counts[label] / n) for label in labels)
    kappa = None if math.isclose(expected, 1.0) else (observed - expected) / (1.0 - expected)
    return Agreement(n=n, observed=observed, kappa=kappa, confusion=confusion)


def kappa_label(kappa: float | None) -> str:
    """Landis & Koch (1977) descriptive bands. Conventions, not laws."""
    if kappa is None:
        return "undefined"
    kappa = round(kappa, 9)  # 0.6000000000000001 is 0.6
    if kappa < 0:
        return "poor"
    for bound, label in ((0.2, "slight"), (0.4, "fair"), (0.6, "moderate"), (0.8, "substantial")):
        if kappa <= bound:
            return label
    return "almost perfect"
