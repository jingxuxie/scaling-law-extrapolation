"""Run the exact-risk pilots used by the paper.

All experiments are laptop-scale.  The script writes figures to paper/figures
and numerical summaries to results.  Run from the repository root with

    PYTHONPATH=src python experiments/run_all.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from scaling_extrapolation import (
    MatchedPair,
    certificate_curve,
    hidden_weight_for_tolerance,
    mixture_risk,
    tail_basis,
)


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "paper" / "figures"
RESULT_DIR = ROOT / "results"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def integer_geomspace(start: int, stop: int, count: int) -> np.ndarray:
    """Return increasing integer sizes approximately uniform on log scale."""
    if start <= 0 or stop < start or count < 2:
        raise ValueError("require 0 < start <= stop and count >= 2")
    values = np.rint(np.geomspace(start, stop, count)).astype(int)
    values[0], values[-1] = start, stop
    if np.unique(values).size != count:
        raise ValueError("requested range is too narrow for unique integer sizes")
    return values.astype(float)


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compute_to_target(
    exponents: list[float], weights: list[float], target_excess_risk: float
) -> float:
    """Solve sum_l w_l phi_alpha_l(M) <= target by log-scale bisection."""
    if target_excess_risk <= 0:
        raise ValueError("target_excess_risk must be positive")
    lower, upper = 1.0, 1e100
    for _ in range(240):
        middle = np.sqrt(lower * upper)
        value = sum(
            weight * float(tail_basis([middle], exponent)[0])
            for exponent, weight in zip(exponents, weights, strict=True)
        )
        if value <= target_excess_risk:
            upper = middle
        else:
            lower = middle
    return upper


def matched_floor_exponent_demo() -> None:
    pair = MatchedPair(floor=0.20, signal=0.08, alpha=0.35, beta=0.55)
    half_span = 0.5
    pilot_t = np.linspace(-half_span, half_span, 9)
    target_t = 6.5
    standard_error = 0.005

    dense_t = np.linspace(-half_span, target_t, 800)
    baseline = pair.baseline(dense_t)
    alternative = pair.alternative(dense_t)
    pilot_gap = pair.gap(pilot_t)
    target_gap = float(pair.gap([target_t])[0])
    kl_upper = float(np.sum(pilot_gap**2) / (2.0 * standard_error**2))

    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    ax.plot(np.exp(dense_t), baseline, label=r"$E+S e^{-\alpha t}$")
    ax.plot(np.exp(dense_t), alternative, linestyle="--", label="matched alternative")
    ax.axvspan(np.exp(-half_span), np.exp(half_span), alpha=0.12, label="pilot range")
    ax.set_xscale("log")
    ax.set_xlabel(r"relative scale $M/M_0$")
    ax.set_ylabel("risk")
    ax.set_title("Value-and-slope matching hides floor--exponent ambiguity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "matched_floor_exponent.pdf")
    fig.savefig(FIGURE_DIR / "matched_floor_exponent.png", dpi=180)
    plt.close(fig)

    write_rows(
        RESULT_DIR / "matched_floor_exponent.csv",
        [
            "alpha",
            "beta",
            "half_log_span",
            "pilot_points",
            "standard_error",
            "max_pilot_gap",
            "target_log_scale",
            "target_gap",
            "gaussian_kl",
        ],
        [
            {
                "alpha": pair.alpha,
                "beta": pair.beta,
                "half_log_span": half_span,
                "pilot_points": pilot_t.size,
                "standard_error": standard_error,
                "max_pilot_gap": float(np.max(pilot_gap)),
                "target_log_scale": target_t,
                "target_gap": target_gap,
                "gaussian_kl": kl_upper,
            }
        ],
    )


def hidden_crossover_demo() -> None:
    alpha = 0.25
    beta = 0.80
    visible_weight = 0.10
    pilot_min = 32.0
    pilot_max = 4096.0
    pilot_points = 9
    standard_error = 0.002
    tolerance = standard_error / (2.0 * np.sqrt(pilot_points))
    hidden_weight = hidden_weight_for_tolerance(pilot_min, alpha, tolerance)
    pilot_sizes = integer_geomspace(int(pilot_min), int(pilot_max), pilot_points)
    pilot_truth = (
        visible_weight * tail_basis(pilot_sizes, beta)
        + hidden_weight * tail_basis(pilot_sizes, alpha)
    )

    def standardized_residual(parameters: np.ndarray) -> np.ndarray:
        weight, exponent = parameters
        fitted = weight * tail_basis(pilot_sizes, float(exponent))
        return (fitted - pilot_truth) / standard_error

    candidates = [
        least_squares(
            standardized_residual,
            x0=np.asarray(start, dtype=float),
            bounds=([0.0, 0.05], [10.0, 2.0]),
            max_nfev=10_000,
        )
        for start in ([0.10, 0.80], [0.10, 0.50], [0.01, 0.20])
    ]
    fit = min(candidates, key=lambda result: float(np.sum(result.fun**2)))
    fitted_weight, fitted_exponent = map(float, fit.x)

    sizes = np.geomspace(pilot_min, 1e10, 800)
    visible = visible_weight * tail_basis(sizes, beta)
    hidden = hidden_weight * tail_basis(sizes, alpha)
    mixture = visible + hidden
    fitted_single_power = fitted_weight * tail_basis(sizes, fitted_exponent)

    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    ax.plot(sizes, visible, label=rf"visible component, $\beta={beta}$")
    ax.plot(sizes, hidden, linestyle=":", label=rf"hidden component, $\alpha={alpha}$")
    ax.plot(sizes, mixture, linestyle="--", label="true two-component tail")
    ax.plot(
        sizes,
        fitted_single_power,
        linestyle="-.",
        label=rf"pilot fit, $\widehat\alpha={fitted_exponent:.3f}$",
    )
    ax.axvspan(pilot_min, pilot_max, alpha=0.12, label="pilot range")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("model size $M$")
    ax.set_ylabel("reducible risk")
    ax.set_title("A below-noise component changes the asymptotic exponent")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "hidden_crossover.pdf")
    fig.savefig(FIGURE_DIR / "hidden_crossover.png", dpi=180)
    plt.close(fig)

    rows: list[dict[str, object]] = []
    for target in [pilot_min, pilot_max, 1e5, 1e7, 1e9]:
        visible_at_target = float(visible_weight * tail_basis([target], beta)[0])
        hidden_at_target = float(hidden_weight * tail_basis([target], alpha)[0])
        rows.append(
            {
                "target_size": target,
                "visible_tail": visible_at_target,
                "hidden_tail": hidden_at_target,
                "hidden_to_visible_ratio": hidden_at_target / visible_at_target,
                "hidden_weight": hidden_weight,
                "pilot_tolerance": tolerance,
            }
        )
    write_rows(
        RESULT_DIR / "hidden_crossover.csv",
        [
            "target_size",
            "visible_tail",
            "hidden_tail",
            "hidden_to_visible_ratio",
            "hidden_weight",
            "pilot_tolerance",
        ],
        rows,
    )

    target_rows: list[dict[str, object]] = []
    for target_risk in [1e-3, 3e-4, 1e-4, 3e-5, 1e-5]:
        true_compute = compute_to_target(
            [alpha, beta], [hidden_weight, visible_weight], target_risk
        )
        fitted_compute = compute_to_target(
            [fitted_exponent], [fitted_weight], target_risk
        )
        target_rows.append(
            {
                "target_excess_risk": target_risk,
                "true_required_size": true_compute,
                "single_power_predicted_size": fitted_compute,
                "underestimation_factor": true_compute / fitted_compute,
                "fitted_weight": fitted_weight,
                "fitted_exponent": fitted_exponent,
                "standardized_rss": float(np.sum(fit.fun**2)),
                "max_absolute_standardized_residual": float(np.max(np.abs(fit.fun))),
            }
        )
    write_rows(
        RESULT_DIR / "single_power_fit.csv",
        [
            "target_excess_risk",
            "true_required_size",
            "single_power_predicted_size",
            "underestimation_factor",
            "fitted_weight",
            "fitted_exponent",
            "standardized_rss",
            "max_absolute_standardized_residual",
        ],
        target_rows,
    )


def certificate_coverage_demo(seed: int = 7, repetitions: int = 250) -> None:
    rng = np.random.default_rng(seed)
    pilot_sizes = integer_geomspace(32, 4096, 9)
    target_sizes = np.asarray([4096.0, 1e5, 1e7, 1e9])
    dictionary = [0.20, 0.25, 0.35, 0.50, 0.80, 1.00]
    true_exponents = [0.25, 0.80]
    true_weights = [0.0002, 0.10]
    floor = 0.20
    standard_error = 0.0002
    delta = 0.10

    pilot_truth = mixture_risk(
        pilot_sizes, floor=floor, exponents=true_exponents, weights=true_weights
    )
    target_truth = mixture_risk(
        target_sizes, floor=floor, exponents=true_exponents, weights=true_weights
    )
    se = np.full_like(pilot_sizes, standard_error)

    covered = np.zeros(target_sizes.size, dtype=int)
    widths = np.zeros((repetitions, target_sizes.size), dtype=float)
    simultaneous = 0
    infeasible = 0
    example: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    for repetition in range(repetitions):
        observations = pilot_truth + rng.normal(scale=se)
        lower, upper, intervals = certificate_curve(
            pilot_sizes,
            observations,
            se,
            target_sizes,
            dictionary,
            delta=delta,
        )
        if not all(item.feasible for item in intervals):
            infeasible += 1
            widths[repetition] = np.nan
            continue
        event = (lower <= target_truth) & (target_truth <= upper)
        covered += event.astype(int)
        simultaneous += int(np.all(event))
        widths[repetition] = upper - lower
        if example is None:
            example = (observations, lower, upper)

    valid = repetitions - infeasible
    if valid <= 0 or example is None:
        raise RuntimeError("all certificate problems were infeasible")

    observations, example_lower, example_upper = example
    curve_targets = np.geomspace(pilot_sizes[-1], 1e9, 80)
    curve_lower, curve_upper, _ = certificate_curve(
        pilot_sizes,
        observations,
        se,
        curve_targets,
        dictionary,
        delta=delta,
    )
    curve_truth = mixture_risk(
        curve_targets, floor=floor, exponents=true_exponents, weights=true_weights
    )

    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    ax.errorbar(
        pilot_sizes,
        observations,
        yerr=se,
        fmt="o",
        capsize=2,
        label="pilot observations",
    )
    ax.plot(curve_targets, curve_truth, label="true risk")
    ax.fill_between(curve_targets, curve_lower, curve_upper, alpha=0.2, label="SCALE-CERT")
    ax.set_xscale("log")
    ax.set_xlabel("model size $M$")
    ax.set_ylabel("risk")
    ax.set_title("Set-valued extrapolation widens beyond the pilot range")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "certificate_demo.pdf")
    fig.savefig(FIGURE_DIR / "certificate_demo.png", dpi=180)
    plt.close(fig)

    rows: list[dict[str, object]] = []
    for index, target in enumerate(target_sizes):
        rows.append(
            {
                "target_size": target,
                "repetitions": repetitions,
                "valid_repetitions": valid,
                "marginal_coverage": covered[index] / valid,
                "mean_width": float(np.nanmean(widths[:, index])),
                "median_width": float(np.nanmedian(widths[:, index])),
                "simultaneous_coverage": simultaneous / valid,
                "simultaneous_coverage_unconditional": simultaneous / repetitions,
                "nominal_coverage": 1.0 - delta,
                "infeasible_runs": infeasible,
                "example_lower": example_lower[index],
                "example_upper": example_upper[index],
                "target_truth": target_truth[index],
            }
        )
    write_rows(
        RESULT_DIR / "certificate_coverage.csv",
        [
            "target_size",
            "repetitions",
            "valid_repetitions",
            "marginal_coverage",
            "mean_width",
            "median_width",
            "simultaneous_coverage",
            "simultaneous_coverage_unconditional",
            "nominal_coverage",
            "infeasible_runs",
            "example_lower",
            "example_upper",
            "target_truth",
        ],
        rows,
    )


def main() -> None:
    matched_floor_exponent_demo()
    hidden_crossover_demo()
    certificate_coverage_demo()
    print(f"Wrote figures to {FIGURE_DIR}")
    print(f"Wrote summaries to {RESULT_DIR}")


if __name__ == "__main__":
    main()
