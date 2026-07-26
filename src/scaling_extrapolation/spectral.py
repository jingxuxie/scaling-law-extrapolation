"""Exact spectral scaling curves for truncated linear regression.

The model is ``y = sum_j theta_j x_j + eps`` with independent coordinates and
predictive energies ``q_j = E[x_j^2] theta_j^2``. A predictor restricted to
the first ``M`` coordinates has population risk ``E[eps^2] + sum_{j>M} q_j``.

Besides evaluating exact Hurwitz-zeta tails, this module supplies rigorous
curvature bounds in the exponent. Those bounds convert a continuous positive
spectral mixture into a conservative finite linear program.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import zeta


FloatArray = NDArray[np.float64]
BasisName = Literal["zeta", "power"]


def _as_float_array(values: ArrayLike) -> FloatArray:
    return np.asarray(values, dtype=float)


def tail_basis(model_size: ArrayLike, exponent: float) -> FloatArray:
    """Return ``phi_alpha(M) = sum_{j>M} j^(-1-alpha)``.

    Integer ``M`` gives the exact spectral tail. Positive real values are
    accepted through the Hurwitz-zeta continuation for smooth plotting.
    """
    if exponent <= 0:
        raise ValueError("exponent must be positive")
    m = _as_float_array(model_size)
    if np.any(m < 0):
        raise ValueError("model_size must be nonnegative")
    return np.asarray(zeta(1.0 + exponent, m + 1.0), dtype=float)


def power_basis(model_size: ArrayLike, exponent: float) -> FloatArray:
    """Return the conventional positive power basis ``M^{-alpha}``."""
    if exponent <= 0:
        raise ValueError("exponent must be positive")
    m = _as_float_array(model_size)
    if np.any(m <= 0):
        raise ValueError("model_size must be positive")
    return np.asarray(m ** (-exponent), dtype=float)


def evaluate_basis(
    model_size: ArrayLike, exponent: float, *, basis: BasisName = "zeta"
) -> FloatArray:
    """Evaluate one of the supported positive scaling-law bases."""
    if basis == "zeta":
        return tail_basis(model_size, exponent)
    if basis == "power":
        return power_basis(model_size, exponent)
    raise ValueError(f"unknown basis: {basis}")


def asymptotic_tail(model_size: ArrayLike, exponent: float) -> FloatArray:
    """Leading approximation ``phi_alpha(M) ~= M^{-alpha}/alpha``."""
    if exponent <= 0:
        raise ValueError("exponent must be positive")
    m = _as_float_array(model_size)
    if np.any(m <= 0):
        raise ValueError("model_size must be positive")
    return np.asarray(m ** (-exponent) / exponent, dtype=float)


def mixture_risk(
    model_size: ArrayLike,
    floor: float,
    exponents: Iterable[float],
    weights: Iterable[float],
    *,
    basis: BasisName = "zeta",
) -> FloatArray:
    """Evaluate a positive finite-mixture risk curve."""
    exponents_arr = np.asarray(tuple(exponents), dtype=float)
    weights_arr = np.asarray(tuple(weights), dtype=float)
    if floor < 0:
        raise ValueError("floor must be nonnegative")
    if exponents_arr.ndim != 1 or weights_arr.ndim != 1:
        raise ValueError("exponents and weights must be one-dimensional")
    if exponents_arr.size != weights_arr.size:
        raise ValueError("exponents and weights must have equal length")
    if np.any(exponents_arr <= 0) or np.any(weights_arr < 0):
        raise ValueError("exponents must be positive and weights nonnegative")

    m = _as_float_array(model_size)
    out = np.full_like(m, fill_value=float(floor), dtype=float)
    for alpha, weight in zip(exponents_arr, weights_arr, strict=True):
        out += weight * evaluate_basis(m, float(alpha), basis=basis)
    return out


def _log_squared_tail_integral(lower: float, exponent: float) -> float:
    """Compute ``int_lower^inf log(x)^2 x^{-1-alpha} dx`` exactly."""
    if lower <= 0 or exponent <= 0:
        raise ValueError("lower and exponent must be positive")
    log_lower = float(np.log(lower))
    alpha = float(exponent)
    return float(
        lower ** (-alpha)
        * (
            log_lower**2 / alpha
            + 2.0 * log_lower / alpha**2
            + 2.0 / alpha**3
        )
    )


def tail_exponent_curvature_upper(
    model_size: ArrayLike, exponent: float
) -> FloatArray:
    r"""Upper-bound ``d^2 phi_alpha(M) / d alpha^2``.

    For integer ``M >= 1``, the exact curvature is
    ``sum_{j>M} (log j)^2 j^{-1-alpha}``. The summand is eventually decreasing;
    we sum its finite pre-asymptotic part and bound the tail by an exact
    integral. The returned value is deterministic and conservative.
    """
    if exponent <= 0:
        raise ValueError("exponent must be positive")
    m_raw = _as_float_array(model_size)
    if np.any(m_raw < 1) or np.any(np.abs(m_raw - np.rint(m_raw)) > 1e-10):
        raise ValueError("zeta curvature bound requires integer model_size >= 1")

    m_values = np.rint(m_raw).astype(int)
    flat = m_values.reshape(-1)
    out = np.empty(flat.size, dtype=float)
    threshold = int(np.ceil(np.exp(2.0 / (1.0 + exponent))))

    for index, m in enumerate(flat):
        split = max(int(m), threshold)
        finite = 0.0
        if split > m:
            j = np.arange(m + 1, split + 1, dtype=float)
            finite = float(np.sum(np.log(j) ** 2 * j ** (-1.0 - exponent)))
        tail = _log_squared_tail_integral(float(split), exponent)
        out[index] = finite + tail

    return out.reshape(m_values.shape)


def power_exponent_curvature(
    model_size: ArrayLike, exponent: float
) -> FloatArray:
    """Return ``d^2 M^{-alpha}/d alpha^2`` exactly."""
    m = _as_float_array(model_size)
    if np.any(m <= 0):
        raise ValueError("model_size must be positive")
    if exponent <= 0:
        raise ValueError("exponent must be positive")
    return np.asarray(np.log(m) ** 2 * m ** (-exponent), dtype=float)


def exponent_curvature_upper(
    model_size: ArrayLike,
    exponent_left: float,
    exponent_right: float | None = None,
    *,
    basis: BasisName = "zeta",
) -> FloatArray:
    """Upper-bound basis curvature over an exponent interval."""
    right = exponent_left if exponent_right is None else exponent_right
    if not (0 < exponent_left <= right):
        raise ValueError("require 0 < exponent_left <= exponent_right")
    if basis == "zeta":
        return tail_exponent_curvature_upper(model_size, exponent_left)
    if basis == "power":
        left_values = power_exponent_curvature(model_size, exponent_left)
        right_values = power_exponent_curvature(model_size, right)
        return np.maximum(left_values, right_values)
    raise ValueError(f"unknown basis: {basis}")


@dataclass(frozen=True)
class MatchedPair:
    """Two floor-plus-power curves matching value and slope at ``t=0``."""

    floor: float
    signal: float
    alpha: float
    beta: float

    def __post_init__(self) -> None:
        if self.floor < 0:
            raise ValueError("floor must be nonnegative")
        if self.signal <= 0:
            raise ValueError("signal must be positive")
        if not (0 < self.alpha < self.beta):
            raise ValueError("require 0 < alpha < beta")

    @property
    def alternative_floor(self) -> float:
        return self.floor + self.signal * (1.0 - self.alpha / self.beta)

    @property
    def alternative_signal(self) -> float:
        return self.alpha * self.signal / self.beta

    def baseline(self, log_scale: ArrayLike) -> FloatArray:
        t = _as_float_array(log_scale)
        return self.floor + self.signal * np.exp(-self.alpha * t)

    def alternative(self, log_scale: ArrayLike) -> FloatArray:
        t = _as_float_array(log_scale)
        return self.alternative_floor + self.alternative_signal * np.exp(
            -self.beta * t
        )

    def gap(self, log_scale: ArrayLike) -> FloatArray:
        return self.alternative(log_scale) - self.baseline(log_scale)


def hidden_weight_for_tolerance(
    pilot_min_size: float, exponent: float, tolerance: float
) -> float:
    """Largest hidden-component weight with gap ``tolerance`` at pilot minimum."""
    if tolerance < 0:
        raise ValueError("tolerance must be nonnegative")
    basis_value = float(tail_basis(np.asarray([pilot_min_size]), exponent)[0])
    if basis_value <= 0:
        raise RuntimeError("tail basis must be positive")
    return tolerance / basis_value
