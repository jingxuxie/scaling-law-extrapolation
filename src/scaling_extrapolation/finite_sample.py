"""Finite-sample regression bridges for spectral risk curves.

For independent Gaussian coordinates, omitted features act as independent
homoskedastic noise for the first ``M`` coordinates. Ordinary least squares
therefore has an exact mean and an exact Hotelling/F distribution for its
population test risk. These identities validate the extrapolation phenomenon
with trained estimators at negligible computational cost.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import f as f_distribution


FloatArray = NDArray[np.float64]


def ols_mean_multiplier(n_samples: int, dimension: int) -> float:
    """Return ``E[R_OLS(M)] / R(M) = (n-1)/(n-M-1)``."""
    if dimension < 0:
        raise ValueError("dimension must be nonnegative")
    if n_samples <= dimension + 1:
        raise ValueError("require n_samples > dimension + 1")
    return float((n_samples - 1) / (n_samples - dimension - 1))


def ols_debias_multiplier(n_samples: int, dimension: int) -> float:
    """Factor making the OLS population risk unbiased for ``R(M)``."""
    return 1.0 / ols_mean_multiplier(n_samples, dimension)


def expected_ols_population_risk(
    oracle_risk: ArrayLike, n_samples: int, dimension: ArrayLike
) -> FloatArray:
    """Evaluate the exact expected OLS population risk coordinatewise."""
    risks = np.asarray(oracle_risk, dtype=float)
    dims = np.asarray(dimension, dtype=int)
    risks, dims = np.broadcast_arrays(risks, dims)
    if np.any(risks < 0):
        raise ValueError("oracle_risk must be nonnegative")
    out = np.empty(risks.shape, dtype=float)
    for index in np.ndindex(risks.shape):
        out[index] = risks[index] * ols_mean_multiplier(n_samples, int(dims[index]))
    return out


def sample_ols_population_risk(
    oracle_risk: ArrayLike,
    n_samples: int,
    dimension: ArrayLike,
    *,
    repetitions: int,
    rng: np.random.Generator,
) -> FloatArray:
    r"""Sample the exact unconditional OLS population-risk distribution.

    For ``nu=n-M+1``,
    ``R_OLS(M) =d R(M)[1 + M/nu F_{M,nu}]``.
    """
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    risks = np.asarray(oracle_risk, dtype=float)
    dims = np.asarray(dimension, dtype=int)
    risks, dims = np.broadcast_arrays(risks, dims)
    if np.any(risks < 0) or np.any(dims < 0):
        raise ValueError("risks and dimensions must be nonnegative")
    if np.any(n_samples <= dims + 1):
        raise ValueError("require n_samples > dimension + 1")

    output = np.empty((repetitions,) + risks.shape, dtype=float)
    for index in np.ndindex(risks.shape):
        m = int(dims[index])
        risk = float(risks[index])
        if m == 0:
            output[(slice(None),) + index] = risk
            continue
        nu = n_samples - m + 1
        draws = f_distribution.rvs(
            dfn=m,
            dfd=nu,
            size=repetitions,
            random_state=rng,
        )
        output[(slice(None),) + index] = risk * (1.0 + (m / nu) * draws)
    return output


@dataclass(frozen=True)
class TrainedCurveSummary:
    dimensions: FloatArray
    oracle_risk: FloatArray
    mean_trained_risk: FloatArray
    standard_error: FloatArray
    debiased_mean: FloatArray
    debiased_standard_error: FloatArray


def simulate_ols_curve(
    dimensions: ArrayLike,
    oracle_risk: ArrayLike,
    *,
    n_samples: int,
    repetitions: int,
    seed: int,
) -> TrainedCurveSummary:
    """Simulate an exact finite-sample OLS curve and its mean debiasing."""
    dims = np.asarray(dimensions, dtype=int).reshape(-1)
    risks = np.asarray(oracle_risk, dtype=float).reshape(-1)
    if dims.size != risks.size:
        raise ValueError("dimensions and oracle_risk must align")
    rng = np.random.default_rng(seed)
    draws = sample_ols_population_risk(
        risks,
        n_samples,
        dims,
        repetitions=repetitions,
        rng=rng,
    )
    means = np.mean(draws, axis=0)
    standard_errors = np.std(draws, axis=0, ddof=1) / np.sqrt(repetitions)
    debias = np.asarray(
        [ols_debias_multiplier(n_samples, int(m)) for m in dims], dtype=float
    )
    return TrainedCurveSummary(
        dimensions=dims.astype(float),
        oracle_risk=risks,
        mean_trained_risk=means,
        standard_error=standard_errors,
        debiased_mean=means * debias,
        debiased_standard_error=standard_errors * debias,
    )


def direct_ols_population_risk_draws(
    *,
    n_samples: int,
    dimension: int,
    oracle_risk: float,
    repetitions: int,
    seed: int,
) -> FloatArray:
    """Generate explicit Gaussian-design OLS draws for an implementation check."""
    if n_samples <= dimension + 1:
        raise ValueError("require n_samples > dimension + 1")
    if oracle_risk < 0 or repetitions <= 0:
        raise ValueError("invalid risk or repetitions")
    rng = np.random.default_rng(seed)
    output = np.empty(repetitions, dtype=float)
    for rep in range(repetitions):
        x = rng.normal(size=(n_samples, dimension))
        residual = rng.normal(scale=np.sqrt(oracle_risk), size=n_samples)
        estimate, *_ = np.linalg.lstsq(x, residual, rcond=None)
        output[rep] = oracle_risk + float(np.dot(estimate, estimate))
    return output
