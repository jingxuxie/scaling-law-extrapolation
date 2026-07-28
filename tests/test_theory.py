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
    omitted_upper = (2_000_000 - 1) ** (-alpha) / alpha
    assert 0.0 <= exact - direct <= omitted_upper * 1.001


def test_matched_pair_matches_value_and_slope() -> None:
    pair = MatchedPair(floor=0.2, signal=0.08, alpha=0.35, beta=0.55)
    assert np.isclose(pair.baseline([0.0])[0], pair.alternative([0.0])[0])
    step = 1e-6
    baseline_slope = (pair.baseline([step])[0] - pair.baseline([-step])[0]) / (2 * step)
    alternative_slope = (pair.alternative([step])[0] - pair.alternative([-step])[0]) / (2 * step)
    assert np.isclose(baseline_slope, alternative_slope, rtol=1e-7, atol=1e-10)


def test_matched_pair_gap_obeys_quadratic_upper_bound() -> None:
    pair = MatchedPair(floor=0.2, signal=0.08, alpha=0.35, beta=0.55)
    t = np.linspace(-1.0, 1.0, 401)
    gap = pair.gap(t)
    delta = pair.beta - pair.alpha
    bound = 0.5 * pair.alpha * pair.signal * delta * np.exp(pair.beta * np.abs(t)) * t**2
    assert np.all(gap >= -1e-14)
    assert np.all(gap <= bound + 1e-14)


def test_matched_pair_gap_obeys_target_lower_bound() -> None:
    pair = MatchedPair(floor=0.2, signal=0.08, alpha=0.35, beta=0.55)
    t = np.linspace(0.0, 5.0, 401)
    gap = pair.gap(t)
    delta = pair.beta - pair.alpha
    bound = 0.5 * pair.alpha * pair.signal * delta * np.exp(-pair.beta * t) * t**2
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
    target_truth = float(mixture_risk([target], floor=0.2, exponents=exponents, weights=[0.03, 0.2])[0])
    interval = certificate_interval(sizes, truth, standard_errors, target, exponents, delta=0.1)
    assert interval.feasible
    assert interval.lower <= target_truth <= interval.upper


def test_zeta_exponent_curvature_bound_is_conservative() -> None:
    from scaling_extrapolation import tail_exponent_curvature_upper

    for model_size in [1, 4, 32, 256]:
        for alpha in [0.1, 0.4, 1.2]:
            indices = np.arange(model_size + 1, 1_000_000, dtype=float)
            partial = float(np.sum(np.log(indices) ** 2 * indices ** (-1.0 - alpha)))
            bound = float(tail_exponent_curvature_upper([model_size], alpha)[0])
            assert partial <= bound * (1.0 + 1e-12)


def test_secant_interpolation_error_obeys_curvature_bound() -> None:
    from scaling_extrapolation import exponent_curvature_upper

    sizes = np.asarray([4.0, 32.0, 256.0])
    left, right = 0.25, 0.40
    width = right - left
    bound = width**2 * exponent_curvature_upper(sizes, left, right, basis="zeta") / 8.0
    for alpha in np.linspace(left, right, 31):
        bary_left = (right - alpha) / width
        secant = bary_left * tail_basis(sizes, left) + (1.0 - bary_left) * tail_basis(sizes, right)
        gap = secant - tail_basis(sizes, float(alpha))
        assert np.all(gap >= -1e-13)
        assert np.all(gap <= bound + 1e-12)


def test_continuous_certificate_contains_off_grid_mixture() -> None:
    from scaling_extrapolation import continuous_certificate_interval

    sizes = np.rint(np.geomspace(16, 4096, 10))
    true_exponents = [0.337, 0.873]
    weights = [0.004, 0.12]
    floor = 0.18
    truth = mixture_risk(sizes, floor, true_exponents, weights)
    target = 1_000_000.0
    target_truth = float(mixture_risk([target], floor, true_exponents, weights)[0])
    interval = continuous_certificate_interval(
        sizes, truth, np.full(sizes.size, 1e-7), target, np.linspace(0.15, 1.2, 22), delta=0.05
    )
    assert interval.feasible
    assert interval.lower <= target_truth <= interval.upper


def test_continuous_discretization_slack_refines_quadratically() -> None:
    from scaling_extrapolation import continuous_cell_matrices

    maxima = []
    widths = []
    for points in [11, 21, 41, 81]:
        grid = np.linspace(0.2, 1.2, points)
        _, slack = continuous_cell_matrices([1_000_000.0], grid, basis="zeta")
        maxima.append(float(np.max(slack)))
        widths.append(float(np.max(np.diff(grid))))
    slope = np.polyfit(np.log(widths), np.log(maxima), 1)[0]
    assert 1.8 <= slope <= 2.2


def test_ols_exact_mean_and_debiasing() -> None:
    from scaling_extrapolation import expected_ols_population_risk, ols_debias_multiplier, ols_mean_multiplier

    oracle = 0.23
    n_samples = 500
    dimension = 40
    expected = float(expected_ols_population_risk([oracle], n_samples, [dimension])[0])
    assert np.isclose(expected, oracle * ols_mean_multiplier(n_samples, dimension))
    assert np.isclose(expected * ols_debias_multiplier(n_samples, dimension), oracle)


def test_exact_ols_sampler_matches_theoretical_mean() -> None:
    from scaling_extrapolation import expected_ols_population_risk, sample_ols_population_risk

    oracle = np.asarray([0.21, 0.17])
    dims = np.asarray([8, 40])
    n_samples = 500
    draws = sample_ols_population_risk(
        oracle, n_samples, dims, repetitions=80_000, rng=np.random.default_rng(4)
    )
    empirical = np.mean(draws, axis=0)
    expected = expected_ols_population_risk(oracle, n_samples, dims)
    assert np.allclose(empirical, expected, rtol=2e-3)


def test_direct_ols_simulation_agrees_with_theory() -> None:
    from scaling_extrapolation import direct_ols_population_risk_draws, expected_ols_population_risk

    n_samples = 160
    dimension = 6
    oracle = 0.2
    direct = direct_ols_population_risk_draws(
        n_samples=n_samples, dimension=dimension, oracle_risk=oracle, repetitions=1200, seed=9
    )
    expected = float(expected_ols_population_risk([oracle], n_samples, [dimension])[0])
    assert np.isclose(np.mean(direct), expected, rtol=0.025)


def test_parametric_baselines_recover_clean_one_power() -> None:
    from scaling_extrapolation import fit_floor_power, fit_two_power

    x = np.geomspace(1.0, 128.0, 12)
    y = 0.2 + 0.3 * x ** (-0.6)
    one = fit_floor_power(x, y)
    two = fit_two_power(x, y)
    target = 1024.0
    truth = 0.2 + 0.3 * target ** (-0.6)
    assert abs(float(one.predict([target])[0]) - truth) < 1e-5
    assert abs(float(two.predict([target])[0]) - truth) < 2e-3


def test_minimum_uniform_radius_is_zero_for_supported_curve() -> None:
    from scaling_extrapolation import minimum_uniform_band_radius

    x = np.geomspace(1.0, 128.0, 10)
    grid = np.linspace(0.2, 1.0, 17)
    y = mixture_risk(x, 0.2, [0.35, 0.8], [0.04, 0.2], basis="power")
    radius = minimum_uniform_band_radius(x, y, grid, basis="power")
    assert radius <= 1e-8


def test_slowest_supported_component_controls_large_scale_slope() -> None:
    sizes = np.asarray([1e8, 1e12])
    excess = mixture_risk(
        sizes,
        floor=0.0,
        exponents=[0.25, 0.90],
        weights=[1e-3, 0.20],
    )
    fitted_slope = -np.log(excess[1] / excess[0]) / np.log(sizes[1] / sizes[0])
    assert abs(fitted_slope - 0.25) < 1e-3


def test_fixed_budget_pilot_designs_follow_effective_span_order() -> None:
    from scipy.stats import norm

    pair = MatchedPair(floor=0.20, signal=0.08, alpha=0.35, beta=0.55)
    sigma = 0.005
    designs = {
        "centered": np.linspace(-0.1, 0.1, 9),
        "uniform": np.linspace(-0.5, 0.5, 9),
        "endpoint": np.asarray([-0.5] * 4 + [0.0] + [0.5] * 4),
    }
    effective_spans = []
    kls = []
    bayes_errors = []
    for design in designs.values():
        gap = pair.gap(design)
        kl = float(np.sum(gap**2) / (2.0 * sigma**2))
        effective_spans.append(float(np.mean(design**4) ** 0.25))
        kls.append(kl)
        bayes_errors.append(float(norm.cdf(-np.sqrt(2.0 * kl) / 2.0)))

    assert effective_spans[0] < effective_spans[1] < effective_spans[2]
    assert kls[0] < kls[1] < kls[2]
    assert bayes_errors[0] > bayes_errors[1] > bayes_errors[2]
