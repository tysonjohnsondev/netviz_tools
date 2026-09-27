from __future__ import annotations

import warnings
from dataclasses import FrozenInstanceError
from typing import Any

import networkx as nx
import numpy as np
import pytest

import netviz_tools as nv
from netviz_tools import InsufficientDataError, UnknownMetricError
from netviz_tools.stats import Distribution, DistributionComparison, _ln_loglik, fit_power_law


def pareto_sample(n: int, alpha: float, xmin: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return xmin * (1.0 - rng.random(n)) ** (-1.0 / (alpha - 1.0))


def test_continuous_recovers_alpha_and_beats_exponential() -> None:
    x = pareto_sample(3000, 2.5, 1.0, seed=0)
    fit = fit_power_law(x, discrete=False, xmin=1.0)
    assert fit.alpha == pytest.approx(2.5, abs=0.08)
    assert fit.vs_exponential.favoured() == "power_law"
    assert fit.n == fit.n_tail == 3000


def test_discrete_recovers_alpha() -> None:
    rng = np.random.default_rng(1)
    x = rng.zipf(2.3, size=4000)
    fit = fit_power_law(x, discrete=True, xmin=1)
    assert fit.discrete
    assert fit.alpha == pytest.approx(2.3, abs=0.06)
    assert fit.vs_exponential.favoured() == "power_law"


def test_lognormal_data_is_not_called_power_law() -> None:
    rng = np.random.default_rng(2)
    x = rng.lognormal(mean=2.0, sigma=0.6, size=4000)
    fit = fit_power_law(x, discrete=False)
    assert fit.vs_lognormal.loglikelihood_ratio < 0
    assert fit.vs_lognormal.favoured() in {"lognormal", "inconclusive"}


def test_exponential_data_favours_exponential() -> None:
    rng = np.random.default_rng(3)
    x = 1.0 + rng.exponential(2.0, size=3000)
    fit = fit_power_law(x, discrete=False, xmin=1.0)
    assert fit.vs_exponential.favoured() == "exponential"
    assert fit.exponential_rate == pytest.approx(0.5, rel=0.05)


def test_matches_powerlaw_package() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import powerlaw  # type: ignore[import-untyped]

        x = pareto_sample(1500, 2.2, 3.0, seed=4) + np.random.default_rng(5).random(1500)
        ours = fit_power_law(x, discrete=False)
        ref = powerlaw.Fit(x, discrete=False, verbose=False)
        r_ln, p_ln = ref.distribution_compare("power_law", "lognormal", normalized_ratio=False)
        r_ex, p_ex = ref.distribution_compare("power_law", "exponential", normalized_ratio=False)
    assert ours.xmin == pytest.approx(ref.xmin)
    assert ours.alpha == pytest.approx(ref.alpha, rel=1e-6)
    # powerlaw uses a one-sided empirical CDF; ours is the two-sided KS statistic.
    assert ours.ks_distance == pytest.approx(ref.D, abs=2e-3)
    assert ours.vs_exponential.loglikelihood_ratio == pytest.approx(r_ex, rel=1e-3)
    assert ours.vs_exponential.p_value == pytest.approx(p_ex, abs=1e-3)
    # Both fit the truncated lognormal numerically. Check that the likelihoods
    # agree for the same parameters, and that our optimum is at least as good.
    tail = x[x >= ours.xmin]
    theirs = _ln_loglik(tail, ours.xmin, False, ref.lognormal.mu, ref.lognormal.sigma).sum()
    assert theirs == pytest.approx(ref.lognormal.loglikelihoods(tail).sum(), rel=1e-9)
    mine = _ln_loglik(tail, ours.xmin, False, ours.lognormal_mu, ours.lognormal_sigma).sum()
    assert mine >= theirs - 1e-6
    assert ours.vs_lognormal.loglikelihood_ratio == pytest.approx(r_ln, abs=0.05)
    assert abs(ours.vs_lognormal.loglikelihood_ratio) < 1.0
    assert p_ln > 0.5
    assert ours.vs_lognormal.favoured() == "inconclusive"


def test_degree_distribution_fit_on_graph(wheat_graph: nx.DiGraph[Any]) -> None:
    quantities: tuple[nv.stats.FitQuantity, ...] = ("degree", "in_degree", "out_degree")
    for quantity in quantities:
        fit = nv.stats.degree_distribution_fit(wheat_graph, quantity)
        assert fit.discrete
        assert fit.quantity == quantity
        assert fit.n_tail >= 10
    strength = nv.stats.degree_distribution_fit(wheat_graph, "out_strength")
    assert not strength.discrete
    assert 1.0 < strength.alpha < 5.0
    assert "vs lognormal" in strength.summary()
    und: nx.Graph[str] = nx.Graph(wheat_graph)
    assert nv.stats.degree_distribution_fit(und, "in_strength").n == len(und)


def test_ccdf_is_one_at_xmin() -> None:
    x = pareto_sample(500, 2.5, 2.0, seed=6)
    for discrete, data in ((False, x), (True, np.ceil(x))):
        fit = fit_power_law(data, discrete=discrete, xmin=2.0)
        dists: tuple[Distribution, ...] = ("power_law", "lognormal", "exponential")
        for dist in dists:
            values = fit.ccdf([2.0, 10.0], dist)
            assert values[0] == pytest.approx(1.0)
            assert values[1] < 1.0
        with pytest.raises(UnknownMetricError):
            fit.ccdf([2.0], "weibull")  # type: ignore[arg-type]


def test_result_is_frozen() -> None:
    fit = fit_power_law(pareto_sample(200, 2.5, 1.0, seed=7), discrete=False)
    with pytest.raises(FrozenInstanceError):
        fit.alpha = 3.0  # type: ignore[misc]


def test_comparison_verdicts() -> None:
    assert DistributionComparison("lognormal", 5.0, 0.01).favoured() == "power_law"
    assert DistributionComparison("lognormal", -5.0, 0.01).favoured() == "lognormal"
    assert DistributionComparison("lognormal", -5.0, 0.5).favoured() == "inconclusive"
    assert DistributionComparison("lognormal", -5.0, float("nan")).favoured() == "inconclusive"


def test_insufficient_data() -> None:
    with pytest.raises(InsufficientDataError, match="at least"):
        fit_power_law([1, 2, 3], discrete=True)
    with pytest.raises(InsufficientDataError, match="equal"):
        fit_power_law([5.0] * 20, discrete=False)


def test_identical_tail_values_give_unit_p_value() -> None:
    # Every point equals xmin except one, so pointwise ratios can be constant.
    data = [1.0] * 30 + [2.0]
    fit = fit_power_law(data, discrete=True, xmin=1.0)
    assert 0.0 <= fit.vs_exponential.p_value <= 1.0


def test_unknown_quantity(wheat_graph: nx.DiGraph[Any]) -> None:
    with pytest.raises(UnknownMetricError):
        nv.stats.degree_distribution_fit(wheat_graph, "closeness")  # type: ignore[arg-type]
