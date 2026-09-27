"""Heavy-tail tests for degree and strength distributions.

A straight line on a log-log plot is weak evidence of a power law. This module
follows Clauset, Shalizi and Newman (2009), "Power-law distributions in
empirical data", SIAM Review 51(4):

1. Estimate the exponent by maximum likelihood for every candidate lower
   bound ``xmin``, and keep the ``xmin`` whose fit has the smallest
   Kolmogorov-Smirnov distance to the data.
2. Fit lognormal and exponential distributions to the same tail.
3. Compare each alternative with the power law using Vuong's normalized
   log-likelihood ratio test.

Degrees are integers and use the discrete likelihoods (Hurwitz zeta for the
power law). Strengths are real-valued and use the continuous likelihoods.

The implementation uses SciPy directly instead of the ``powerlaw`` package,
which would add matplotlib and mpmath as dependencies. The test suite checks
the continuous estimates against ``powerlaw``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias, get_args

import networkx as nx
import numpy as np
import numpy.typing as npt
from scipy import optimize, special, stats

from netviz_tools.errors import InsufficientDataError, UnknownMetricError

__all__ = [
    "DistributionComparison",
    "FitQuantity",
    "PowerLawFit",
    "degree_distribution_fit",
    "fit_power_law",
]

FitQuantity: TypeAlias = Literal[
    "degree", "in_degree", "out_degree", "strength", "in_strength", "out_strength"
]
"""Which per-node quantity to fit. Degrees are discrete; strengths continuous."""

Distribution: TypeAlias = Literal["power_law", "lognormal", "exponential"]

FloatArray: TypeAlias = npt.NDArray[np.float64]


@dataclass(frozen=True)
class DistributionComparison:
    """Result of a likelihood-ratio test of the power law against one alternative.

    Attributes
    ----------
    alternative
        Name of the alternative distribution.
    loglikelihood_ratio
        Vuong's ``R``: the summed log-likelihood of the power law minus that of
        the alternative, over the tail. Positive values favour the power law.
    p_value
        Two-sided p-value for ``R != 0``. A large p-value means the data cannot
        tell the two distributions apart.
    """

    alternative: str
    loglikelihood_ratio: float
    p_value: float

    def favoured(self, significance: float = 0.1) -> str:
        """Return the better-supported distribution, or ``"inconclusive"``.

        Parameters
        ----------
        significance
            Threshold for the p-value. Clauset et al. use 0.1.

        Returns
        -------
        str
            ``"power_law"``, the alternative's name, or ``"inconclusive"``.
        """
        if not self.p_value < significance:
            return "inconclusive"
        return "power_law" if self.loglikelihood_ratio > 0 else self.alternative


@dataclass(frozen=True)
class PowerLawFit:
    """A power-law fit to the tail of a distribution, with model comparisons.

    Attributes
    ----------
    quantity
        What was fitted, for example ``"degree"``.
    discrete
        Whether discrete likelihoods were used.
    alpha
        Estimated exponent of ``p(x) ~ x**-alpha`` for ``x >= xmin``.
    xmin
        Lower bound of the power-law tail.
    n
        Number of positive observations.
    n_tail
        Number of observations at or above ``xmin``.
    ks_distance
        Kolmogorov-Smirnov distance between the tail and the fitted power law.
    vs_lognormal, vs_exponential
        Likelihood-ratio tests against the two alternatives.
    lognormal_mu, lognormal_sigma
        Parameters of the lognormal fitted to the tail.
    exponential_rate
        Rate of the exponential fitted to the tail.
    values
        The observations, sorted ascending. Excluded from ``repr`` and equality.
    """

    quantity: str
    discrete: bool
    alpha: float
    xmin: float
    n: int
    n_tail: int
    ks_distance: float
    vs_lognormal: DistributionComparison
    vs_exponential: DistributionComparison
    lognormal_mu: float
    lognormal_sigma: float
    exponential_rate: float
    values: tuple[float, ...] = field(repr=False, compare=False)

    def ccdf(self, x: npt.ArrayLike, distribution: Distribution = "power_law") -> FloatArray:
        """Evaluate a fitted model's complementary CDF, conditional on ``x >= xmin``.

        Parameters
        ----------
        x
            Points at or above ``xmin``.
        distribution
            ``"power_law"``, ``"lognormal"`` or ``"exponential"``.

        Returns
        -------
        numpy.ndarray
            ``P(X >= x | X >= xmin)``.
        """
        xs = np.asarray(x, dtype=float)
        if distribution == "power_law":
            return _pl_sf(xs, self.alpha, self.xmin, self.discrete)
        if distribution == "lognormal":
            return _ln_sf(xs, self.lognormal_mu, self.lognormal_sigma, self.xmin, self.discrete)
        if distribution == "exponential":
            return np.exp(-self.exponential_rate * (xs - self.xmin))
        raise UnknownMetricError(f"unknown distribution {distribution!r}")

    def summary(self) -> str:
        """Return a one-paragraph plain-text summary of the fit."""
        parts = [
            f"{self.quantity}: alpha={self.alpha:.3f}, xmin={self.xmin:g}, "
            f"tail {self.n_tail} of {self.n}, KS={self.ks_distance:.3f}."
        ]
        for c in (self.vs_lognormal, self.vs_exponential):
            parts.append(
                f"vs {c.alternative}: R={c.loglikelihood_ratio:.2f}, p={c.p_value:.3f} "
                f"({c.favoured()})."
            )
        return " ".join(parts)


# ---------------------------------------------------------------------------
# Likelihoods and survival functions


def _pl_sf(x: FloatArray, alpha: float, xmin: float, discrete: bool) -> FloatArray:
    if discrete:
        return np.asarray(special.zeta(alpha, x) / special.zeta(alpha, xmin), dtype=float)
    return np.asarray((x / xmin) ** (1.0 - alpha), dtype=float)


def _ln_sf(x: FloatArray, mu: float, sigma: float, xmin: float, discrete: bool) -> FloatArray:
    shift = 0.5 if discrete else 0.0
    num = stats.norm.logsf((np.log(x - shift) - mu) / sigma)
    den = stats.norm.logsf((math.log(xmin - shift) - mu) / sigma)
    return np.asarray(np.exp(num - den), dtype=float)


def _pl_fit(tail: FloatArray, xmin: float, discrete: bool) -> tuple[float, FloatArray]:
    logs = np.log(tail)
    if not discrete:
        alpha = 1.0 + tail.size / float(np.sum(logs - math.log(xmin)))
        ll = math.log(alpha - 1.0) - math.log(xmin) - alpha * (logs - math.log(xmin))
        return alpha, np.asarray(ll, dtype=float)
    sum_logs = float(logs.sum())
    n = tail.size

    def nll(a: float) -> float:
        return float(n * math.log(special.zeta(a, xmin)) + a * sum_logs)

    res = optimize.minimize_scalar(nll, bounds=(1.0 + 1e-6, 20.0), method="bounded")
    alpha = float(res.x)
    ll = -alpha * logs - math.log(special.zeta(alpha, xmin))
    return alpha, np.asarray(ll, dtype=float)


def _exp_fit(tail: FloatArray, xmin: float, discrete: bool) -> tuple[float, FloatArray]:
    excess = max(float(tail.mean()) - xmin, 1e-12)
    if discrete:
        lam = math.log1p(1.0 / excess)
        ll = math.log(-math.expm1(-lam)) - lam * (tail - xmin)
    else:
        lam = 1.0 / excess
        ll = math.log(lam) - lam * (tail - xmin)
    return lam, np.asarray(ll, dtype=float)


def _ln_loglik(
    tail: FloatArray, xmin: float, discrete: bool, mu: float, sigma: float
) -> FloatArray:
    if discrete:
        hi = stats.norm.logsf((np.log(tail - 0.5) - mu) / sigma)
        lo = stats.norm.logsf((np.log(tail + 0.5) - mu) / sigma)
        # log(sf(a) - sf(b)) computed stably from the two log survival values.
        mass = hi + np.log(-np.expm1(np.minimum(lo - hi, -1e-300)))
        norm = stats.norm.logsf((math.log(xmin - 0.5) - mu) / sigma)
        return np.asarray(mass - norm, dtype=float)
    logs = np.log(tail)
    z = (logs - mu) / sigma
    norm = stats.norm.logsf((math.log(xmin) - mu) / sigma)
    ll = -logs - math.log(sigma) - 0.5 * math.log(2 * math.pi) - 0.5 * z**2 - norm
    return np.asarray(ll, dtype=float)


def _ln_fit(tail: FloatArray, xmin: float, discrete: bool) -> tuple[float, float, FloatArray]:
    logs = np.log(tail)
    start = np.array([float(logs.mean()), math.log(max(float(logs.std()), 0.1))])

    def nll(params: FloatArray) -> float:
        mu, log_sigma = float(params[0]), float(params[1])
        value = -float(np.sum(_ln_loglik(tail, xmin, discrete, mu, math.exp(log_sigma))))
        return value if math.isfinite(value) else 1e300

    res = optimize.minimize(
        nll, start, method="Nelder-Mead", options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 4000}
    )
    mu, sigma = float(res.x[0]), math.exp(float(res.x[1]))
    return mu, sigma, _ln_loglik(tail, xmin, discrete, mu, sigma)


def _ks(tail: FloatArray, alpha: float, xmin: float, discrete: bool) -> float:
    n = tail.size
    if discrete:
        uniq, counts = np.unique(tail, return_counts=True)
        emp = np.cumsum(counts) / n
        model = 1.0 - _pl_sf(uniq + 1.0, alpha, xmin, True)
        return float(np.max(np.abs(emp - model)))
    model = 1.0 - _pl_sf(tail, alpha, xmin, False)
    i = np.arange(1, n + 1)
    return float(np.max(np.maximum(np.abs(i / n - model), np.abs((i - 1) / n - model))))


def _vuong(ll_a: FloatArray, ll_b: FloatArray, name: str) -> DistributionComparison:
    diff = ll_a - ll_b
    r = float(diff.sum())
    sd = float(diff.std())
    if sd == 0.0 or not math.isfinite(sd):
        return DistributionComparison(name, r, 1.0)
    p = float(math.erfc(abs(r) / (math.sqrt(2.0 * diff.size) * sd)))
    return DistributionComparison(name, r, p)


# ---------------------------------------------------------------------------
# Public API


def fit_power_law(
    values: npt.ArrayLike,
    *,
    discrete: bool,
    xmin: float | None = None,
    min_tail: int = 10,
    quantity: str = "values",
) -> PowerLawFit:
    """Fit a power law to the tail of a sample and compare it with alternatives.

    Parameters
    ----------
    values
        Observations. Non-positive values are ignored.
    discrete
        Use discrete likelihoods. Values are then expected to be integers.
    xmin
        Fix the lower bound instead of choosing it by KS distance.
    min_tail
        Smallest tail size considered when choosing ``xmin``.
    quantity
        Label stored on the result.

    Returns
    -------
    PowerLawFit
        The fit and its comparisons with lognormal and exponential models.

    Raises
    ------
    InsufficientDataError
        If fewer than ``min_tail`` positive observations are available, or all
        tail values are equal.
    """
    data = np.sort(np.asarray(values, dtype=float))
    data = data[np.isfinite(data) & (data > 0)]
    if data.size < max(min_tail, 2):
        raise InsufficientDataError(
            f"need at least {max(min_tail, 2)} positive observations, got {data.size}"
        )
    if xmin is not None:
        candidates = np.array([float(xmin)])
    else:
        uniq = np.unique(data)
        # Keep candidates that leave at least `min_tail` points in the tail.
        # The smallest value always qualifies, because data.size >= min_tail.
        candidates = np.array([u for u in uniq if np.sum(data >= u) >= min_tail])
    best: tuple[float, float, float] | None = None
    for cand in candidates:
        tail = data[data >= cand]
        if tail.size < 2 or np.all(tail == tail[0]):
            continue
        alpha, _ = _pl_fit(tail, float(cand), discrete)
        d = _ks(tail, alpha, float(cand), discrete)
        if best is None or d < best[0]:
            best = (d, float(cand), alpha)
    if best is None:
        raise InsufficientDataError("all tail values are equal; the exponent is undefined")
    ks, x0, _ = best
    tail = data[data >= x0]
    alpha, ll_pl = _pl_fit(tail, x0, discrete)
    rate, ll_exp = _exp_fit(tail, x0, discrete)
    mu, sigma, ll_ln = _ln_fit(tail, x0, discrete)
    return PowerLawFit(
        quantity=quantity,
        discrete=discrete,
        alpha=alpha,
        xmin=x0,
        n=int(data.size),
        n_tail=int(tail.size),
        ks_distance=ks,
        vs_lognormal=_vuong(ll_pl, ll_ln, "lognormal"),
        vs_exponential=_vuong(ll_pl, ll_exp, "exponential"),
        lognormal_mu=mu,
        lognormal_sigma=sigma,
        exponential_rate=rate,
        values=tuple(float(v) for v in data),
    )


def degree_distribution_fit(
    g: nx.Graph[Any],
    quantity: FitQuantity = "degree",
    *,
    xmin: float | None = None,
    min_tail: int = 10,
) -> PowerLawFit:
    """Test whether a graph's degree or strength distribution has a power-law tail.

    Parameters
    ----------
    g
        A graph; strengths use the edge attribute ``"weight"``.
    quantity
        Per-node quantity to fit; see :data:`FitQuantity`.
    xmin
        Fix the lower bound of the tail instead of estimating it.
    min_tail
        Smallest tail size considered when estimating ``xmin``.

    Returns
    -------
    PowerLawFit
        Exponent, ``xmin``, goodness of fit, and likelihood-ratio tests
        against lognormal and exponential tails.

    Raises
    ------
    UnknownMetricError
        If ``quantity`` is not supported.
    InsufficientDataError
        If the graph is too small for a tail fit.

    Examples
    --------
    >>> import netviz_tools as nv
    >>> g = nv.build_graph(nv.datasets.faostat.load_sample(items="Wheat", years=2021))
    >>> fit = nv.stats.degree_distribution_fit(g, "out_strength")
    >>> fit.vs_lognormal.alternative
    'lognormal'
    """
    if quantity not in get_args(FitQuantity):
        raise UnknownMetricError(
            f"unknown quantity {quantity!r}; choose from {list(get_args(FitQuantity))}"
        )
    weighted = quantity.endswith("strength")
    weight = "weight" if weighted else None
    mode = quantity.split("_")[0] if "_" in quantity else "all"
    if g.is_directed() and mode == "in":
        pairs = g.in_degree(weight=weight)  # type: ignore[attr-defined]
    elif g.is_directed() and mode == "out":
        pairs = g.out_degree(weight=weight)  # type: ignore[attr-defined]
    else:
        pairs = g.degree(weight=weight)
    values = [float(v) for _, v in pairs]
    return fit_power_law(
        values, discrete=not weighted, xmin=xmin, min_tail=min_tail, quantity=quantity
    )
