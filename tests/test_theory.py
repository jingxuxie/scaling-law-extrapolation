from __future__ import annotations

import numpy as np

from scaling_extrapolation import (
    MatchedPair,
    certificate_interval,
    hidden_weight_for_tolerance,
    mixture_risk,
    tail_basis,
)


def test_tail_basis_matches_direct_sum() -> None:
    m = 20
    alpha = 0.7
    direct = np.sum(np.arange(m + 1, 2_000_000, dtype=float) ** (-1.0 - alpha))
    exact = float(tail_basis([m], alpha)[0])
    # The finite direct sum omits only a tiny positive tail.
    omitted_upper = (2_000_000 - 1) ** (-alpha) / alpha
    assert 0.0 <= exact - direct <= omitted_upper * 1.001


def test_matched_pair_matches_value_and_slope() -> None:
    pair = MatchedPair(floor=0.2, signal=0.08, alpha=0.35, beta=0.55)
    assert np.isclose(pair.baseline([0.0])[0], pair.alternative([0.0])[0])
    step = 1e-6
    baseline_slope = (pair.baseline([step])[0] - pair.baseline([-step])[0]) / (2 * step)
    alternative_slope = (
        pair.alternative([step])[0] - pair.alternative([-step])[0]
    ) / (2 * step)
    assert np.isclose(baseline_slope, alternative_slope, rtol=1e-7, atol=1e-10)


def test_matched_pair_gap_obeys_quadratic_upper_bound() -> None:
    pair = MatchedPair(floor=0.2, signal=0.08, alpha=0.35, beta=0.55)
    t = np.linspace(-1.0, 1.0, 401)
    gap = pair.gap(t)
    delta = pair.beta - pair.alpha
    bound = (
        0.5
        * pair.alpha
        * pair.signal
        * delta
        * np.exp(pair.beta * np.abs(t))
        * t**2
    )
    assert np.all(gap >= -1e-14)
    assert np.all(gap <= bound + 1e-14)


def test_matched_pair_gap_obeys_target_lower_bound() -> None:
    pair = MatchedPair(floor=0.2, signal=0.08, alpha=0.35, beta=0.55)
    t = np.linspace(0.0, 5.0, 401)
    gap = pair.gap(t)
    delta = pair.beta - pair.alpha
    bound = (
        0.5
        * pair.alpha
        * pair.signal
        * delta
        * np.exp(-pair.beta * t)
        * t**2
    )
    assert np.all(gap + 1e-14 >= bound)


def test_exact_power_predictive_energies_telescope() -> None:
    alpha = 0.6
    amplitude = 0.7
    model_size = 137
    indices = np.arange(model_size + 1, 2_000_000, dtype=float)
    energies = amplitude * (indices ** (-alpha) - (indices + 1.0) ** (-alpha))
    finite_tail = float(np.sum(energies))
    exact_tail = amplitude * (model_size + 1.0) ** (-alpha)
    omitted = amplitude * 2_000_000.0 ** (-alpha)
    assert np.isclose(finite_tail + omitted, exact_tail, rtol=1e-11, atol=1e-13)


def test_hidden_weight_saturates_pilot_tolerance() -> None:
    m_min = 32.0
    alpha = 0.25
    tolerance = 1e-3
    weight = hidden_weight_for_tolerance(m_min, alpha, tolerance)
    gap = weight * float(tail_basis([m_min], alpha)[0])
    assert np.isclose(gap, tolerance)


def test_certificate_contains_correctly_specified_curve() -> None:
    sizes = np.rint(np.geomspace(32, 4096, 9))
    exponents = [0.25, 0.8]
    truth = mixture_risk(sizes, floor=0.2, exponents=exponents, weights=[0.03, 0.2])
    standard_errors = np.full_like(sizes, 1e-4)
    target = 65536.0
    target_truth = float(
        mixture_risk([target], floor=0.2, exponents=exponents, weights=[0.03, 0.2])[0]
    )
    interval = certificate_interval(
        sizes,
        truth,
        standard_errors,
        target,
        exponents,
        delta=0.1,
    )
    assert interval.feasible
    assert interval.lower <= target_truth <= interval.upper
