"""Statistics: checked against scipy/numpy as oracles, plus invariants."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats as sps

from evalkit.stats import (
    cluster_bootstrap_delta,
    cohen_kappa,
    fisher_exact,
    holm_bonferroni,
    kappa_label,
    percentile,
    smallest_detectable_drop,
    wilson_interval,
)

counts = st.integers(min_value=1, max_value=400).flatmap(
    lambda n: st.tuples(st.integers(min_value=0, max_value=n), st.just(n))
)


@given(counts, st.sampled_from([0.8, 0.9, 0.95, 0.99]))
def test_wilson_matches_scipy(kn: tuple[int, int], confidence: float) -> None:
    k, n = kn
    ours = wilson_interval(k, n, confidence)
    ref = sps.binomtest(k, n).proportion_ci(confidence_level=confidence, method="wilson")
    assert ours.low == pytest.approx(ref.low, abs=1e-9)
    assert ours.high == pytest.approx(ref.high, abs=1e-9)


@given(counts)
def test_wilson_invariants(kn: tuple[int, int]) -> None:
    k, n = kn
    ci = wilson_interval(k, n)
    assert 0.0 <= ci.low <= ci.estimate <= ci.high <= 1.0
    assert ci.estimate == k / n
    if k == 0:
        assert ci.low == 0.0
    if k == n:
        assert ci.high == 1.0


def test_wilson_edge_cases() -> None:
    assert wilson_interval(0, 0).low == 0.0
    assert wilson_interval(0, 0).high == 1.0
    with pytest.raises(ValueError, match="invalid counts"):
        wilson_interval(5, 3)
    with pytest.raises(ValueError, match="confidence"):
        wilson_interval(1, 2, confidence=1.0)


cells = st.integers(min_value=0, max_value=40)


@settings(max_examples=300)
@given(cells, cells, cells, cells, st.sampled_from(["two-sided", "less", "greater"]))
def test_fisher_matches_scipy(a: int, b: int, c: int, d: int, alternative: str) -> None:
    ours = fisher_exact(a, b, c, d, alternative)  # type: ignore[arg-type]
    ref = sps.fisher_exact([[a, b], [c, d]], alternative=alternative).pvalue
    assert ours == pytest.approx(ref, rel=1e-9, abs=1e-12)


def test_fisher_known_values() -> None:
    # 10/10 vs 0/10: two-sided p = 2 / C(20, 10)
    assert fisher_exact(10, 0, 0, 10) == pytest.approx(2 / math.comb(20, 10))
    # identical rows are never significant
    assert fisher_exact(7, 3, 7, 3) == 1.0
    # degenerate margins
    assert fisher_exact(0, 0, 3, 4) == 1.0
    assert fisher_exact(5, 0, 5, 0) == 1.0


def test_holm_known_example() -> None:
    assert holm_bonferroni([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert holm_bonferroni([]) == []


@given(st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=1, max_size=30))
def test_holm_invariants(ps: list[float]) -> None:
    adj = holm_bonferroni(ps)
    assert all(p <= q <= 1.0 for p, q in zip(ps, adj, strict=True))
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    ranked = [adj[i] for i in order]
    assert ranked == sorted(ranked)  # monotone in the raw p-value order
    assert min(adj) == pytest.approx(min(1.0, len(ps) * min(ps)))


@given(
    st.lists(st.floats(min_value=-1e6, max_value=1e6), min_size=1, max_size=50),
    st.floats(min_value=0.0, max_value=1.0),
)
def test_percentile_matches_numpy(values: list[float], q: float) -> None:
    assert percentile(values, q) == pytest.approx(float(np.percentile(values, q * 100)), abs=1e-6)


def test_cluster_bootstrap() -> None:
    same = cluster_bootstrap_delta([(10, 10, 10, 10), (0, 10, 0, 10)] * 4)
    assert (same.estimate, same.low, same.high) == (0.0, 0.0, 0.0)
    pairs = [(10, 10, 0, 10)] + [(9, 10, 9, 10)] * 7
    a = cluster_bootstrap_delta(pairs, seed=7)
    assert a == cluster_bootstrap_delta(pairs, seed=7)  # deterministic for a seed
    assert a.estimate == pytest.approx(-1.0 / 8)
    assert a.low < a.estimate < a.high
    assert a.low < 0.0 <= a.high  # one outlier case among eight is not a suite trend
    # The bug this design fixes: a single case going 5/5 -> 4/5 must not look certain.
    single = cluster_bootstrap_delta([(5, 5, 4, 5)])
    assert single.estimate == pytest.approx(-0.2)
    assert single.high == 0.0
    consistent = cluster_bootstrap_delta([(9, 10, 6, 10)] * 12)
    assert consistent.high < 0.0
    with pytest.raises(ValueError, match="no paired"):
        cluster_bootstrap_delta([])
    with pytest.raises(ValueError, match="invalid counts"):
        cluster_bootstrap_delta([(3, 2, 1, 1)])


@given(
    st.lists(
        st.tuples(st.integers(0, 12), st.integers(0, 12)).map(lambda t: (min(t), 12, max(t), 12)),
        min_size=1,
        max_size=6,
    ),
    st.integers(0, 10_000),
)
@settings(max_examples=40, deadline=None)
def test_cluster_bootstrap_brackets_point_estimate(
    pairs: list[tuple[int, int, int, int]], seed: int
) -> None:
    ci = cluster_bootstrap_delta(pairs, resamples=400, seed=seed)
    assert -1.0 <= ci.low <= ci.high <= 1.0
    # Resampled means are centred on the point estimate, so the interval brackets it.
    assert ci.low - 1e-9 <= ci.estimate <= ci.high + 1e-9


def test_kappa_wikipedia_example() -> None:
    # 50 proposals: yes/yes 20, yes/no 5, no/yes 10, no/no 15 -> kappa 0.4
    pairs = [("y", "y")] * 20 + [("y", "n")] * 5 + [("n", "y")] * 10 + [("n", "n")] * 15
    agreement = cohen_kappa(pairs)
    assert agreement.observed == pytest.approx(0.7)
    assert agreement.kappa == pytest.approx(0.4)
    assert agreement.confusion == {"n": {"n": 15, "y": 10}, "y": {"n": 5, "y": 20}}


def test_kappa_edge_cases() -> None:
    assert cohen_kappa([("a", "a"), ("b", "b")]).kappa == pytest.approx(1.0)
    assert cohen_kappa([("a", "a")] * 5).kappa is None  # undefined: one label only
    assert kappa_label(None) == "undefined"
    assert kappa_label(0.487) == "moderate"
    assert kappa_label(-0.1) == "poor"
    assert kappa_label(0.95) == "almost perfect"
    with pytest.raises(ValueError, match="no rated"):
        cohen_kappa([])


@given(st.lists(st.tuples(st.sampled_from("pf"), st.sampled_from("pf")), min_size=1, max_size=60))
def test_kappa_symmetric_and_bounded(pairs: list[tuple[str, str]]) -> None:
    k1 = cohen_kappa(pairs).kappa
    k2 = cohen_kappa([(b, a) for a, b in pairs]).kappa
    if k1 is None:
        assert k2 is None
    else:
        assert k2 == pytest.approx(k1)
        assert -1.0 - 1e-9 <= k1 <= 1.0 + 1e-9


def test_smallest_detectable_drop() -> None:
    # 10 vs 10 trials, Holm over 8 cases: a perfect case must fall to <= 30%.
    assert smallest_detectable_drop(10, 10, comparisons=8) == pytest.approx(0.7)
    # 3 vs 3: even 3/3 -> 0/3 has p = 0.1, so nothing is detectable.
    assert smallest_detectable_drop(3, 3) is None
    assert smallest_detectable_drop(0, 5) is None
