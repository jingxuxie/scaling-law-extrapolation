"""Exact spectral scaling curves for truncated linear regression.

The model is y = sum_j theta_j x_j + eps with independent coordinates and
predictive energies q_j = E[x_j^2] theta_j^2.  A predictor restricted to the
first M coordinates has population risk E[eps^2] + sum_{j>M} q_j.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import zeta


FloatArray = NDArray[np.float64]


def tail_basis(model_size: ArrayLike, exponent: float) -> FloatArray:
    """Return phi_alpha(M) = sum_{j>M} j^(-1-alpha).

    Parameters
    ----------
    model_size:
        Positive model sizes. Integer values correspond exactly to truncation
        dimensions; real values are accepted for smooth plotting.
    exponent:
        Positive tail exponent alpha.
    """
    if exponent <= 0:
        raise ValueError("exponent must be positive")
    m = np.asarray(model_size, dtype=float)
    if np.any(m < 0):
        raise ValueError("model_size must be nonnegative")
    return np.asarray(zeta(1.0 + exponent, m + 1.0), dtype=float)


def asymptotic_tail(model_size: ArrayLike, exponent: float) -> FloatArray:
    """Leading asymptotic approximation phi_alpha(M) ~= M^(-alpha)/alpha."""
    if exponent <= 0:
        raise ValueError("exponent must be positive")
    m = np.asarray(model_size, dtype=float)
    if np.any(m <= 0):
        raise ValueError("model_size must be positive")
    return np.asarray(m ** (-exponent) / exponent, dtype=float)


def mixture_risk(
    model_size: ArrayLike,
    floor: float,
    exponents: Iterable[float],
    weights: Iterable[float],
) -> FloatArray:
    """Evaluate an exact positive spectral-mixture risk curve."""
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

    m = np.asarray(model_size, dtype=float)
    out = np.full_like(m, fill_value=float(floor), dtype=float)
    for alpha, weight in zip(exponents_arr, weights_arr, strict=True):
        out += weight * tail_basis(m, float(alpha))
    return out


@dataclass(frozen=True)
class MatchedPair:
    """Two floor-plus-power curves matching value and slope at t=0."""

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
        t = np.asarray(log_scale, dtype=float)
        return self.floor + self.signal * np.exp(-self.alpha * t)

    def alternative(self, log_scale: ArrayLike) -> FloatArray:
        t = np.asarray(log_scale, dtype=float)
        return self.alternative_floor + self.alternative_signal * np.exp(
            -self.beta * t
        )

    def gap(self, log_scale: ArrayLike) -> FloatArray:
        return self.alternative(log_scale) - self.baseline(log_scale)


def hidden_weight_for_tolerance(
    pilot_min_size: float, exponent: float, tolerance: float
) -> float:
    """Largest hidden-component weight with gap `tolerance` at pilot minimum."""
    if tolerance < 0:
        raise ValueError("tolerance must be nonnegative")
    basis = float(tail_basis(np.asarray([pilot_min_size]), exponent)[0])
    if basis <= 0:
        raise RuntimeError("tail basis must be positive")
    return tolerance / basis
