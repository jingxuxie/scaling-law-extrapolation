"""Deterministic parametric baselines for scaling-law extrapolation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares


FloatArray = NDArray[np.float64]
ModelName = Literal["one_power", "two_power"]


@dataclass(frozen=True)
class ParametricFit:
    model: ModelName
    x_scale: float
    parameters: FloatArray
    success: bool
    rss: float
    message: str

    def predict(self, x: ArrayLike) -> FloatArray:
        x_norm = np.asarray(x, dtype=float) / self.x_scale
        if np.any(x_norm <= 0):
            raise ValueError("x must be positive")
        if self.model == "one_power":
            floor, amplitude, exponent = self.parameters
            return floor + amplitude * x_norm ** (-exponent)
        floor, amp_1, amp_2, exponent_1, exponent_2 = self.parameters
        return (
            floor
            + amp_1 * x_norm ** (-exponent_1)
            + amp_2 * x_norm ** (-exponent_2)
        )


def _prepare(
    x: ArrayLike, y: ArrayLike, standard_errors: ArrayLike | None
) -> tuple[FloatArray, FloatArray, FloatArray, float]:
    x_arr = np.asarray(x, dtype=float).reshape(-1)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    if x_arr.size != y_arr.size or x_arr.size < 3:
        raise ValueError("x and y must align and contain at least three points")
    if np.any(x_arr <= 0) or np.any(~np.isfinite(y_arr)):
        raise ValueError("x must be positive and y finite")
    if standard_errors is None:
        se = np.ones_like(y_arr)
    else:
        se = np.asarray(standard_errors, dtype=float).reshape(-1)
        if se.size != y_arr.size or np.any(se <= 0):
            raise ValueError("standard_errors must align and be positive")
    x_scale = float(np.min(x_arr))
    return x_arr / x_scale, y_arr, se, x_scale


def fit_floor_power(
    x: ArrayLike,
    y: ArrayLike,
    standard_errors: ArrayLike | None = None,
    *,
    initial: ArrayLike | None = None,
    multistart: bool = True,
) -> ParametricFit:
    """Fit ``E + A (x/x_min)^(-alpha)`` by constrained least squares."""
    x_norm, y_arr, se, x_scale = _prepare(x, y, standard_errors)
    y_scale = max(float(np.max(np.abs(y_arr))), 1e-8)
    lower = np.asarray([0.0, 0.0, 0.01])
    upper = np.asarray(
        [max(2.0 * float(np.max(y_arr)), 2.0 * y_scale, 1e-6), 20.0 * y_scale, 4.0]
    )

    def residual(parameters: FloatArray) -> FloatArray:
        floor, amplitude, exponent = parameters
        return (floor + amplitude * x_norm ** (-exponent) - y_arr) / se

    starts: list[FloatArray] = []
    if initial is not None:
        starts.append(np.asarray(initial, dtype=float))
    floor_guesses = [0.0, max(0.0, float(np.min(y_arr)) * 0.5), max(0.0, float(np.min(y_arr)) * 0.9)]
    exponent_guesses = [0.1, 0.25, 0.5, 1.0, 2.0] if multistart else [0.5]
    for floor in floor_guesses if multistart else floor_guesses[:1]:
        amplitude = max(float(y_arr[0] - floor), 1e-8)
        for exponent in exponent_guesses:
            starts.append(np.asarray([floor, amplitude, exponent], dtype=float))
    results = [
        least_squares(
            residual,
            x0=np.clip(start, lower + 1e-10, upper - 1e-10),
            bounds=(lower, upper),
            max_nfev=4000,
        )
        for start in starts
    ]
    best = min(results, key=lambda result: float(np.sum(result.fun**2)))
    return ParametricFit(
        model="one_power",
        x_scale=x_scale,
        parameters=np.asarray(best.x, dtype=float),
        success=bool(best.success),
        rss=float(np.sum(best.fun**2)),
        message=str(best.message),
    )


def fit_two_power(
    x: ArrayLike,
    y: ArrayLike,
    standard_errors: ArrayLike | None = None,
    *,
    initial: ArrayLike | None = None,
    multistart: bool = True,
) -> ParametricFit:
    """Fit a positive two-power mixture with an unknown nonnegative floor."""
    x_norm, y_arr, se, x_scale = _prepare(x, y, standard_errors)
    y_scale = max(float(np.max(np.abs(y_arr))), 1e-8)
    lower = np.asarray([0.0, 0.0, 0.0, 0.01, 0.01])
    upper = np.asarray(
        [max(2.0 * float(np.max(y_arr)), 2.0 * y_scale, 1e-6), 20.0 * y_scale, 20.0 * y_scale, 4.0, 4.0]
    )

    def residual(parameters: FloatArray) -> FloatArray:
        floor, amp_1, amp_2, exponent_1, exponent_2 = parameters
        prediction = floor + amp_1 * x_norm ** (-exponent_1) + amp_2 * x_norm ** (-exponent_2)
        return (prediction - y_arr) / se

    starts: list[FloatArray] = []
    if initial is not None:
        starts.append(np.asarray(initial, dtype=float))
    floors = [0.0, max(0.0, float(np.min(y_arr)) * 0.8)] if multistart else [0.0]
    pairs = [(0.1, 0.8), (0.2, 1.5), (0.5, 2.0), (0.1, 0.3)] if multistart else [(0.1, 0.8)]
    for floor in floors:
        amplitude = max(float(y_arr[0] - floor), 1e-8)
        for exponent_1, exponent_2 in pairs:
            starts.append(np.asarray([floor, 0.75 * amplitude, 0.25 * amplitude, exponent_1, exponent_2]))
    results = [
        least_squares(
            residual,
            x0=np.clip(start, lower + 1e-10, upper - 1e-10),
            bounds=(lower, upper),
            max_nfev=7000,
        )
        for start in starts
    ]
    best = min(results, key=lambda result: float(np.sum(result.fun**2)))
    parameters = np.asarray(best.x, dtype=float)
    if parameters[3] > parameters[4]:
        parameters = parameters[[0, 2, 1, 4, 3]]
    return ParametricFit(
        model="two_power",
        x_scale=x_scale,
        parameters=parameters,
        success=bool(best.success),
        rss=float(np.sum(best.fun**2)),
        message=str(best.message),
    )


def parametric_bootstrap_interval(
    x: ArrayLike,
    y: ArrayLike,
    standard_errors: ArrayLike,
    target_x: float,
    *,
    model: ModelName = "one_power",
    repetitions: int = 100,
    delta: float = 0.1,
    seed: int = 0,
) -> tuple[float, float, float, ParametricFit]:
    """Parametric-bootstrap interval conditional on the fitted family."""
    if repetitions <= 1 or not (0 < delta < 1):
        raise ValueError("invalid repetitions or delta")
    x_arr = np.asarray(x, dtype=float).reshape(-1)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    se = np.asarray(standard_errors, dtype=float).reshape(-1)
    fit_function = fit_floor_power if model == "one_power" else fit_two_power
    fitted = fit_function(x_arr, y_arr, se, multistart=True)
    fitted_values = fitted.predict(x_arr)
    point = float(fitted.predict([target_x])[0])
    rng = np.random.default_rng(seed)
    predictions: list[float] = []
    for _ in range(repetitions):
        sample = fitted_values + rng.normal(scale=se)
        refit = fit_function(x_arr, sample, se, initial=fitted.parameters, multistart=False)
        if refit.success:
            prediction = float(refit.predict([target_x])[0])
            if np.isfinite(prediction):
                predictions.append(prediction)
    if len(predictions) < max(10, repetitions // 3):
        return point, float("nan"), float("nan"), fitted
    lower, upper = np.quantile(predictions, [delta / 2.0, 1.0 - delta / 2.0])
    return point, float(lower), float(upper), fitted


def laplace_floor_power_interval(
    x: ArrayLike,
    y: ArrayLike,
    standard_errors: ArrayLike,
    target_x: float,
    *,
    repetitions: int = 4000,
    delta: float = 0.1,
    seed: int = 0,
    prior_log_sd: float = 2.0,
) -> tuple[float, float, float]:
    """Weak-prior Laplace interval for a positive floor-plus-power law."""
    x_arr = np.asarray(x, dtype=float).reshape(-1)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    se = np.asarray(standard_errors, dtype=float).reshape(-1)
    if not (x_arr.size == y_arr.size == se.size) or np.any(se <= 0):
        raise ValueError("inputs must align and standard errors be positive")
    x_scale = float(np.min(x_arr))
    x_norm = x_arr / x_scale
    target_norm = float(target_x / x_scale)
    base = fit_floor_power(x_arr, y_arr, se)
    epsilon = max(1e-10, 1e-8 * float(np.max(np.abs(y_arr))))
    start = np.log(np.maximum(base.parameters, epsilon))
    prior_mean = np.asarray([
        np.log(max(float(np.min(y_arr)) * 0.8, epsilon)),
        np.log(max(float(y_arr[0] - np.min(y_arr) * 0.8), epsilon)),
        np.log(0.5),
    ])

    def augmented_residual(log_parameters: FloatArray) -> FloatArray:
        floor, amplitude, exponent = np.exp(log_parameters)
        prediction = floor + amplitude * x_norm ** (-exponent)
        return np.concatenate(((prediction - y_arr) / se, (log_parameters - prior_mean) / prior_log_sd))

    result = least_squares(
        augmented_residual,
        x0=start,
        bounds=(np.asarray([-30.0, -30.0, -6.0]), np.asarray([10.0, 20.0, 2.0])),
        max_nfev=5000,
    )
    map_log = np.asarray(result.x, dtype=float)
    covariance = np.linalg.pinv(result.jac.T @ result.jac, hermitian=True)
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    root = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0.0)))
    rng = np.random.default_rng(seed)
    draws = map_log + rng.normal(size=(repetitions, 3)) @ root.T
    draws = np.clip(draws, [-30.0, -30.0, -6.0], [10.0, 20.0, 2.0])
    parameters = np.exp(draws)
    predictions = parameters[:, 0] + parameters[:, 1] * target_norm ** (-parameters[:, 2])
    point_parameters = np.exp(map_log)
    point = float(point_parameters[0] + point_parameters[1] * target_norm ** (-point_parameters[2]))
    lower, upper = np.quantile(predictions, [delta / 2.0, 1.0 - delta / 2.0])
    return point, float(lower), float(upper)
