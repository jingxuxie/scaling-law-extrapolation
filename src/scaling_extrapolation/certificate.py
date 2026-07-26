"""Set-valued extrapolation certificates for positive spectral mixtures.

``certificate_interval`` handles a finite exponent dictionary.
``continuous_certificate_interval`` handles every positive mixture whose
exponents lie in a declared interval. It represents each exponent cell by
barycentric endpoint mass and adds a rigorous curvature slack, so the true
exponents need not lie on the numerical grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import linprog
from scipy.stats import norm

from .spectral import BasisName, evaluate_basis, exponent_curvature_upper


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class CertificateInterval:
    """One target-specific partial-identification interval."""

    target_size: float
    lower: float
    upper: float
    feasible: bool
    status: str
    discretization_lower_slack: float = 0.0
    discretization_upper_slack: float = 0.0

    @property
    def width(self) -> float:
        return self.upper - self.lower

    @property
    def bounded(self) -> bool:
        return bool(np.isfinite(self.lower) and np.isfinite(self.upper))


def _validate_grid(exponents: Iterable[float]) -> FloatArray:
    grid = np.asarray(tuple(exponents), dtype=float)
    if grid.ndim != 1 or grid.size == 0 or np.any(grid <= 0):
        raise ValueError("exponents must be a nonempty positive sequence")
    if grid.size > 1 and np.any(np.diff(grid) <= 0):
        raise ValueError("exponents must be strictly increasing")
    return grid


def design_matrix(
    model_sizes: ArrayLike,
    exponents: Iterable[float],
    *,
    basis: BasisName = "zeta",
) -> FloatArray:
    """Build ``[1, phi_alpha1(M), ..., phi_alphaL(M)]``."""
    m = np.asarray(model_sizes, dtype=float).reshape(-1)
    grid = _validate_grid(exponents)
    if m.size == 0:
        raise ValueError("at least one model size is required")
    columns = [np.ones_like(m)]
    columns.extend(evaluate_basis(m, float(a), basis=basis) for a in grid)
    return np.column_stack(columns)


def simultaneous_gaussian_multiplier(delta: float, n_observations: int) -> float:
    """Bonferroni multiplier giving simultaneous coverage at least ``1-delta``."""
    if not (0 < delta < 1):
        raise ValueError("delta must lie in (0, 1)")
    if n_observations <= 0:
        raise ValueError("n_observations must be positive")
    return float(norm.ppf(1.0 - delta / (2.0 * n_observations)))


def gaussian_observation_bands(
    observations: ArrayLike,
    standard_errors: ArrayLike,
    *,
    delta: float,
    misspecification_radius: ArrayLike | float = 0.0,
) -> tuple[FloatArray, FloatArray]:
    """Construct simultaneous two-sided Gaussian bands."""
    y = np.asarray(observations, dtype=float).reshape(-1)
    se = np.asarray(standard_errors, dtype=float).reshape(-1)
    if y.size != se.size:
        raise ValueError("observations and standard_errors must align")
    if np.any(se <= 0):
        raise ValueError("standard_errors must be strictly positive")
    radius = np.broadcast_to(np.asarray(misspecification_radius, dtype=float), y.shape)
    if np.any(radius < 0):
        raise ValueError("misspecification_radius must be nonnegative")
    z = simultaneous_gaussian_multiplier(delta, y.size)
    half_width = z * se + radius
    return y - half_width, y + half_width


def _solve_interval_lp(
    lower_objective: FloatArray,
    upper_objective: FloatArray,
    a_ub: FloatArray,
    b_ub: FloatArray,
    *,
    target_size: float,
    nonnegative_lower: bool = True,
) -> CertificateInterval:
    """Solve lower/upper LPs and handle empty or unbounded confidence sets."""
    n_variables = lower_objective.size
    bounds = [(0.0, None)] * n_variables
    feasibility = linprog(
        c=np.zeros(n_variables),
        A_ub=a_ub,
        b_ub=b_ub,
        bounds=bounds,
        method="highs",
    )
    if not feasibility.success:
        return CertificateInterval(
            target_size=target_size,
            lower=float("nan"),
            upper=float("nan"),
            feasible=False,
            status=f"confidence set empty: {feasibility.message}",
        )

    lower_result = linprog(
        c=lower_objective,
        A_ub=a_ub,
        b_ub=b_ub,
        bounds=bounds,
        method="highs",
    )
    if lower_result.status == 3:
        lower = 0.0 if nonnegative_lower else float("-inf")
        lower_status = "lower objective unbounded; clipped by nonnegative risk"
    elif lower_result.success:
        lower = float(lower_result.fun)
        if nonnegative_lower:
            lower = max(0.0, lower)
        lower_status = "lower optimal"
    else:
        return CertificateInterval(
            target_size=target_size,
            lower=float("nan"),
            upper=float("nan"),
            feasible=False,
            status=f"lower LP failed: {lower_result.message}",
        )

    upper_result = linprog(
        c=-upper_objective,
        A_ub=a_ub,
        b_ub=b_ub,
        bounds=bounds,
        method="highs",
    )
    if upper_result.status == 3:
        upper = float("inf")
        upper_status = "upper objective unbounded"
    elif upper_result.success:
        upper = float(-upper_result.fun)
        upper_status = "upper optimal"
    else:
        return CertificateInterval(
            target_size=target_size,
            lower=float("nan"),
            upper=float("nan"),
            feasible=False,
            status=f"upper LP failed: {upper_result.message}",
        )

    return CertificateInterval(
        target_size=target_size,
        lower=lower,
        upper=upper,
        feasible=True,
        status=f"{lower_status}; {upper_status}",
    )


def certificate_interval_from_bands(
    model_sizes: ArrayLike,
    lower_bands: ArrayLike,
    upper_bands: ArrayLike,
    target_size: float,
    exponents: Iterable[float],
    *,
    basis: BasisName = "zeta",
) -> CertificateInterval:
    """Finite-dictionary certificate from deterministic pilot bands."""
    m = np.asarray(model_sizes, dtype=float).reshape(-1)
    lower = np.asarray(lower_bands, dtype=float).reshape(-1)
    upper = np.asarray(upper_bands, dtype=float).reshape(-1)
    if not (m.size == lower.size == upper.size):
        raise ValueError("model_sizes and bands must align")
    if np.any(lower > upper):
        raise ValueError("lower bands cannot exceed upper bands")
    if target_size <= 0:
        raise ValueError("target_size must be positive")

    grid = _validate_grid(exponents)
    matrix = design_matrix(m, grid, basis=basis)
    target = design_matrix([target_size], grid, basis=basis)[0]
    a_ub = np.vstack((matrix, -matrix))
    b_ub = np.concatenate((upper, -lower))
    return _solve_interval_lp(target, target, a_ub, b_ub, target_size=target_size)


def certificate_interval(
    model_sizes: ArrayLike,
    observations: ArrayLike,
    standard_errors: ArrayLike,
    target_size: float,
    exponents: Iterable[float],
    *,
    delta: float = 0.1,
    misspecification_radius: ArrayLike | float = 0.0,
    basis: BasisName = "zeta",
) -> CertificateInterval:
    """Compute the finite-dictionary LP interval at one target."""
    lower, upper = gaussian_observation_bands(
        observations,
        standard_errors,
        delta=delta,
        misspecification_radius=misspecification_radius,
    )
    return certificate_interval_from_bands(
        model_sizes, lower, upper, target_size, exponents, basis=basis
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
    basis: BasisName = "zeta",
) -> tuple[FloatArray, FloatArray, list[CertificateInterval]]:
    """Evaluate finite-dictionary certificate bounds at many targets."""
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
            basis=basis,
        )
        for target in targets
    ]
    lower = np.asarray([item.lower for item in intervals], dtype=float)
    upper = np.asarray([item.upper for item in intervals], dtype=float)
    return lower, upper, intervals


def continuous_cell_matrices(
    model_sizes: ArrayLike,
    exponent_grid: Iterable[float],
    *,
    basis: BasisName = "zeta",
) -> tuple[FloatArray, FloatArray]:
    r"""Return nominal and interpolation-slack matrices for exponent cells.

    Variables are ``(E, u_0, v_0, ..., u_{L-1}, v_{L-1})``. Cell ``l``
    represents arbitrary positive measure on ``[a_l,a_{l+1}]`` by splitting
    its mass barycentrically between the endpoints. Convexity gives
    ``R(M) <= D(M) theta`` and ``D(M) theta-R(M) <= G(M) theta``.
    """
    m = np.asarray(model_sizes, dtype=float).reshape(-1)
    grid = _validate_grid(exponent_grid)
    if grid.size < 2:
        raise ValueError("continuous certificate needs at least two grid points")
    if m.size == 0:
        raise ValueError("at least one model size is required")

    nominal_columns: list[FloatArray] = [np.ones_like(m)]
    slack_columns: list[FloatArray] = [np.zeros_like(m)]
    for left, right in zip(grid[:-1], grid[1:]):
        width = float(right - left)
        curvature = exponent_curvature_upper(
            m, float(left), float(right), basis=basis
        )
        error = (width**2 / 8.0) * curvature
        nominal_columns.extend(
            [
                evaluate_basis(m, float(left), basis=basis),
                evaluate_basis(m, float(right), basis=basis),
            ]
        )
        slack_columns.extend([error, error])
    return np.column_stack(nominal_columns), np.column_stack(slack_columns)


def continuous_certificate_interval_from_bands(
    model_sizes: ArrayLike,
    lower_bands: ArrayLike,
    upper_bands: ArrayLike,
    target_size: float,
    exponent_grid: Iterable[float],
    *,
    basis: BasisName = "zeta",
) -> CertificateInterval:
    r"""Certify every positive mixture supported on the grid interval.

    This is an outer approximation to the continuous measure problem, not a
    plug-in grid approximation. If the true exponent measure is supported on
    ``[grid[0],grid[-1]]`` and the true pilot risks lie in the supplied bands,
    the interval covers the target even when all true exponents are off-grid.
    """
    m = np.asarray(model_sizes, dtype=float).reshape(-1)
    lower = np.asarray(lower_bands, dtype=float).reshape(-1)
    upper = np.asarray(upper_bands, dtype=float).reshape(-1)
    if not (m.size == lower.size == upper.size):
        raise ValueError("model_sizes and bands must align")
    if np.any(lower > upper):
        raise ValueError("lower bands cannot exceed upper bands")
    if target_size <= 0:
        raise ValueError("target_size must be positive")

    nominal, slack = continuous_cell_matrices(m, exponent_grid, basis=basis)
    target_nominal, target_slack = continuous_cell_matrices(
        [target_size], exponent_grid, basis=basis
    )
    d_star = target_nominal[0]
    g_star = target_slack[0]

    # For the mapped endpoint-split coefficients theta:
    # lower <= R <= D theta and D theta-G theta <= R <= upper.
    a_ub = np.vstack((-nominal, nominal - slack))
    b_ub = np.concatenate((-lower, upper))
    interval = _solve_interval_lp(
        d_star - g_star,
        d_star,
        a_ub,
        b_ub,
        target_size=target_size,
    )
    return CertificateInterval(
        target_size=interval.target_size,
        lower=interval.lower,
        upper=interval.upper,
        feasible=interval.feasible,
        status=interval.status,
        discretization_lower_slack=float(np.max(g_star)),
        discretization_upper_slack=0.0,
    )


def continuous_certificate_interval(
    model_sizes: ArrayLike,
    observations: ArrayLike,
    standard_errors: ArrayLike,
    target_size: float,
    exponent_grid: Iterable[float],
    *,
    delta: float = 0.1,
    misspecification_radius: ArrayLike | float = 0.0,
    basis: BasisName = "zeta",
) -> CertificateInterval:
    """Gaussian-band wrapper for the continuous-exponent certificate."""
    lower, upper = gaussian_observation_bands(
        observations,
        standard_errors,
        delta=delta,
        misspecification_radius=misspecification_radius,
    )
    return continuous_certificate_interval_from_bands(
        model_sizes, lower, upper, target_size, exponent_grid, basis=basis
    )


def continuous_certificate_curve(
    model_sizes: ArrayLike,
    observations: ArrayLike,
    standard_errors: ArrayLike,
    target_sizes: ArrayLike,
    exponent_grid: Iterable[float],
    *,
    delta: float = 0.1,
    misspecification_radius: ArrayLike | float = 0.0,
    basis: BasisName = "zeta",
) -> tuple[FloatArray, FloatArray, list[CertificateInterval]]:
    """Evaluate continuous-exponent certificate bounds at many targets."""
    targets = np.asarray(target_sizes, dtype=float).reshape(-1)
    lower_bands, upper_bands = gaussian_observation_bands(
        observations,
        standard_errors,
        delta=delta,
        misspecification_radius=misspecification_radius,
    )
    intervals = [
        continuous_certificate_interval_from_bands(
            model_sizes,
            lower_bands,
            upper_bands,
            float(target),
            exponent_grid,
            basis=basis,
        )
        for target in targets
    ]
    lower = np.asarray([item.lower for item in intervals], dtype=float)
    upper = np.asarray([item.upper for item in intervals], dtype=float)
    return lower, upper, intervals


def minimum_uniform_band_radius(
    model_sizes: ArrayLike,
    observations: ArrayLike,
    exponent_grid: Iterable[float],
    *,
    basis: BasisName = "power",
) -> float:
    """Smallest uniform pilot radius making the continuous class feasible.

    This diagnostic is useful when public curves have no independent standard
    errors. It does not by itself provide frequentist coverage.
    """
    m = np.asarray(model_sizes, dtype=float).reshape(-1)
    y = np.asarray(observations, dtype=float).reshape(-1)
    if m.size != y.size:
        raise ValueError("model_sizes and observations must align")
    nominal, slack = continuous_cell_matrices(m, exponent_grid, basis=basis)
    n_theta = nominal.shape[1]
    a_ub = np.vstack(
        (
            np.column_stack((-nominal, -np.ones(m.size))),
            np.column_stack((nominal - slack, -np.ones(m.size))),
        )
    )
    b_ub = np.concatenate((-y, y))
    objective = np.zeros(n_theta + 1)
    objective[-1] = 1.0
    result = linprog(
        c=objective,
        A_ub=a_ub,
        b_ub=b_ub,
        bounds=[(0.0, None)] * (n_theta + 1),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"minimum-radius LP failed: {result.message}")
    return float(result.x[-1])
