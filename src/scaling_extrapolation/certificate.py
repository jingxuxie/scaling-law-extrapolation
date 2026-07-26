"""Linear-programming extrapolation certificates for spectral mixtures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import linprog
from scipy.stats import norm

from .spectral import tail_basis


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class CertificateInterval:
    target_size: float
    lower: float
    upper: float
    feasible: bool
    status: str

    @property
    def width(self) -> float:
        return self.upper - self.lower


def design_matrix(model_sizes: ArrayLike, exponents: Iterable[float]) -> FloatArray:
    """Build [1, phi_alpha1(M), ..., phi_alphaL(M)] for exact tail bases."""
    m = np.asarray(model_sizes, dtype=float).reshape(-1)
    alpha = np.asarray(tuple(exponents), dtype=float)
    if m.size == 0:
        raise ValueError("at least one model size is required")
    if alpha.size == 0 or np.any(alpha <= 0):
        raise ValueError("exponents must be a nonempty positive sequence")
    columns = [np.ones_like(m)]
    columns.extend(tail_basis(m, float(a)) for a in alpha)
    return np.column_stack(columns)


def simultaneous_gaussian_multiplier(delta: float, n_observations: int) -> float:
    """Bonferroni multiplier giving simultaneous coverage at least 1-delta."""
    if not (0 < delta < 1):
        raise ValueError("delta must lie in (0, 1)")
    if n_observations <= 0:
        raise ValueError("n_observations must be positive")
    return float(norm.ppf(1.0 - delta / (2.0 * n_observations)))


def _feasible_constraints(
    observations: FloatArray,
    standard_errors: FloatArray,
    matrix: FloatArray,
    delta: float,
    misspecification_radius: ArrayLike | float,
) -> tuple[FloatArray, FloatArray]:
    n = observations.size
    z = simultaneous_gaussian_multiplier(delta, n)
    rho = np.broadcast_to(np.asarray(misspecification_radius, dtype=float), (n,))
    if np.any(rho < 0):
        raise ValueError("misspecification_radius must be nonnegative")
    bands = z * standard_errors + rho
    # |y - X theta| <= bands becomes X theta <= y+bands and
    # -X theta <= -y+bands.
    a_ub = np.vstack((matrix, -matrix))
    b_ub = np.concatenate((observations + bands, -observations + bands))
    return a_ub, b_ub


def certificate_interval(
    model_sizes: ArrayLike,
    observations: ArrayLike,
    standard_errors: ArrayLike,
    target_size: float,
    exponents: Iterable[float],
    *,
    delta: float = 0.1,
    misspecification_radius: ArrayLike | float = 0.0,
) -> CertificateInterval:
    """Compute the sharp LP partial-identification interval at one target.

    The fitted class is E + sum_l w_l phi_{alpha_l}(M), with E,w_l >= 0.
    The confidence set is the intersection of nonnegativity with simultaneous
    Gaussian observation bands.  If the class is correctly specified, the
    returned interval covers the true target risk with probability at least
    1-delta.
    """
    m = np.asarray(model_sizes, dtype=float).reshape(-1)
    y = np.asarray(observations, dtype=float).reshape(-1)
    se = np.asarray(standard_errors, dtype=float).reshape(-1)
    if not (m.size == y.size == se.size):
        raise ValueError("model_sizes, observations, and standard_errors must align")
    if np.any(se <= 0):
        raise ValueError("standard_errors must be strictly positive")
    if target_size <= 0:
        raise ValueError("target_size must be positive")

    alpha = tuple(float(a) for a in exponents)
    x = design_matrix(m, alpha)
    x_star = design_matrix(np.asarray([target_size]), alpha)[0]
    a_ub, b_ub = _feasible_constraints(
        y, se, x, delta, misspecification_radius
    )
    bounds = [(0.0, None)] * x.shape[1]

    lower_result = linprog(
        c=x_star,
        A_ub=a_ub,
        b_ub=b_ub,
        bounds=bounds,
        method="highs",
    )
    if not lower_result.success:
        return CertificateInterval(
            target_size=target_size,
            lower=float("nan"),
            upper=float("nan"),
            feasible=False,
            status=f"lower LP failed: {lower_result.message}",
        )

    upper_result = linprog(
        c=-x_star,
        A_ub=a_ub,
        b_ub=b_ub,
        bounds=bounds,
        method="highs",
    )
    if not upper_result.success:
        return CertificateInterval(
            target_size=target_size,
            lower=float("nan"),
            upper=float("nan"),
            feasible=False,
            status=f"upper LP failed: {upper_result.message}",
        )

    return CertificateInterval(
        target_size=target_size,
        lower=float(lower_result.fun),
        upper=float(-upper_result.fun),
        feasible=True,
        status="optimal",
    )


def certificate_curve(
    model_sizes: ArrayLike,
    observations: ArrayLike,
    standard_errors: ArrayLike,
    target_sizes: ArrayLike,
    exponents: Iterable[float],
    *,
    delta: float = 0.1,
    misspecification_radius: ArrayLike | float = 0.0,
) -> tuple[FloatArray, FloatArray, list[CertificateInterval]]:
    """Evaluate certificate bounds on a sequence of target sizes."""
    targets = np.asarray(target_sizes, dtype=float).reshape(-1)
    intervals = [
        certificate_interval(
            model_sizes,
            observations,
            standard_errors,
            float(target),
            exponents,
            delta=delta,
            misspecification_radius=misspecification_radius,
        )
        for target in targets
    ]
    lower = np.asarray([item.lower for item in intervals], dtype=float)
    upper = np.asarray([item.upper for item in intervals], dtype=float)
    return lower, upper, intervals
