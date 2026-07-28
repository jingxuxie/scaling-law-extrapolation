#!/usr/bin/env python3
"""Reproduce the submission-level theorem and benchmark evidence.

Each stage can run independently. ``--stage all`` launches stages in fresh
subprocesses, preventing long optimizer sweeps from contaminating one another.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import subprocess
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 15,
        "axes.titlesize": 16,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from scaling_extrapolation import (
    continuous_cell_matrices,
    continuous_certificate_interval,
    continuous_certificate_interval_from_bands,
    fit_floor_power,
    fit_two_power,
    laplace_floor_power_interval,
    minimum_uniform_band_radius,
    mixture_risk,
    parametric_bootstrap_interval,
    simulate_ols_curve,
)


RESULTS = ROOT / "results"
FIGURES = ROOT / "paper" / "figures"
PUBLIC = ROOT / "data" / "public"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)
PUBLIC.mkdir(parents=True, exist_ok=True)

PINNED_NSLPFN = "10a5a60911a2a22ef453d8b4ddf1408b6a6075d9"
PUBLIC_URLS = {
    "benchmark.lang.csv": f"https://raw.githubusercontent.com/DongWooLee-Eli/nslpfn/{PINNED_NSLPFN}/data/benchmark.lang.csv",
    "benchmark.vision.csv": f"https://raw.githubusercontent.com/DongWooLee-Eli/nslpfn/{PINNED_NSLPFN}/data/benchmark.vision.csv",
}
PUBLIC_SHA256 = {
    "benchmark.lang.csv": "63c346db00a66b692e1ea9a0f71a2f2b267e55f74be5b863a92e199dffbf0983",
    "benchmark.vision.csv": "23b00692d71e58aa5a2a941e831d57688e0977fdcf5f5125ff7b64283b5d9e83",
}


def stable_seed(*parts: object) -> int:
    """Return a deterministic 32-bit seed independent of PYTHONHASHSEED."""
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty result: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def integer_geomspace(start: int, stop: int, count: int) -> np.ndarray:
    values = np.unique(np.rint(np.geomspace(start, stop, count)).astype(int))
    if values.size != count:
        raise RuntimeError("geometric design produced duplicate dimensions")
    return values.astype(float)


def continuous_refinement() -> None:
    sizes = integer_geomspace(16, 4096, 10)
    exponents = [0.337, 0.873]
    weights = [0.004, 0.12]
    floor = 0.18
    target = 1_000_000.0
    standard_error = 2e-5
    truth = mixture_risk(sizes, floor, exponents, weights)
    target_truth = float(mixture_risk([target], floor, exponents, weights)[0])

    rows: list[dict[str, object]] = []
    for points in [11, 21, 41, 81, 161]:
        grid = np.linspace(0.15, 1.2, points)
        interval = continuous_certificate_interval(
            sizes,
            truth,
            np.full(sizes.size, standard_error),
            target,
            grid,
            delta=0.10,
        )
        _, slack = continuous_cell_matrices([target], grid, basis="zeta")
        rows.append(
            {
                "grid_points": points,
                "max_grid_width": float(np.max(np.diff(grid))),
                "max_target_secant_slack_per_unit_mass": float(np.max(slack)),
                "lower": interval.lower,
                "upper": interval.upper,
                "width": interval.width,
                "target_truth": target_truth,
                "covered": int(interval.lower <= target_truth <= interval.upper),
                "feasible": int(interval.feasible),
            }
        )
    write_csv(RESULTS / "continuous_refinement.csv", rows)

    widths = np.asarray([float(row["max_grid_width"]) for row in rows])
    slacks = np.asarray([float(row["max_target_secant_slack_per_unit_mass"]) for row in rows])
    interval_widths = np.asarray([float(row["width"]) for row in rows])
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.loglog(widths, slacks, "o-", label="secant slack per unit mass")
    ax.loglog(widths, interval_widths, "s--", label="certificate width")
    ax.loglog(
        widths,
        slacks[0] * (widths / widths[0]) ** 2,
        ":",
        label="quadratic",
    )
    ax.set_xlabel("maximum exponent-cell width")
    ax.set_ylabel("correction or interval width")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=2,
        columnspacing=0.8,
        handlelength=1.5,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.78))
    fig.savefig(FIGURES / "continuous_refinement.pdf")
    fig.savefig(FIGURES / "continuous_refinement.png", dpi=180)
    plt.close(fig)


def finite_sample_ols() -> None:
    dimensions = np.asarray([16, 24, 32, 48, 64, 96, 128, 192, 256])
    oracle = mixture_risk(dimensions, 0.02, [0.35, 0.90], [0.004, 0.12])
    n_samples = 8000
    repetitions = int(os.environ.get("OLS_REPETITIONS", "5000"))
    summary = simulate_ols_curve(
        dimensions,
        oracle,
        n_samples=n_samples,
        repetitions=repetitions,
        seed=17,
    )
    target_dimension = 256.0
    pilot = dimensions <= 96
    interval = continuous_certificate_interval(
        summary.dimensions[pilot],
        summary.debiased_mean[pilot],
        summary.debiased_standard_error[pilot],
        target_dimension,
        np.linspace(0.15, 1.20, 43),
        delta=0.10,
    )

    rows: list[dict[str, object]] = []
    for index, dimension in enumerate(dimensions):
        rows.append(
            {
                "dimension": int(dimension),
                "n_samples": n_samples,
                "repetitions": repetitions,
                "oracle_risk": float(summary.oracle_risk[index]),
                "mean_ols_risk": float(summary.mean_trained_risk[index]),
                "ols_standard_error": float(summary.standard_error[index]),
                "debiased_mean": float(summary.debiased_mean[index]),
                "debiased_standard_error": float(summary.debiased_standard_error[index]),
                "relative_debias_error": float(summary.debiased_mean[index] / summary.oracle_risk[index] - 1.0),
                "pilot": int(pilot[index]),
                "target_certificate_lower": interval.lower if dimension == target_dimension else "",
                "target_certificate_upper": interval.upper if dimension == target_dimension else "",
                "target_covered": int(interval.lower <= summary.oracle_risk[index] <= interval.upper) if dimension == target_dimension else "",
            }
        )
    write_csv(RESULTS / "finite_sample_ols.csv", rows)

    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.plot(dimensions, oracle, "o-", label="population risk")
    ax.plot(
        dimensions,
        summary.mean_trained_risk,
        "s--",
        label="mean finite-sample OLS",
    )
    ax.errorbar(
        dimensions,
        summary.debiased_mean,
        yerr=summary.debiased_standard_error,
        fmt="^:",
        capsize=2,
        label="debiased OLS",
    )
    ax.axvline(96, linestyle=":", label=r"last pilot ($M=96$)")
    target_index = int(np.flatnonzero(dimensions == target_dimension)[0])
    target_oracle = float(summary.oracle_risk[target_index])
    ax.errorbar(
        [target_dimension],
        [target_oracle],
        yerr=[
            [target_oracle - interval.lower],
            [interval.upper - target_oracle],
        ],
        fmt="D",
        color="black",
        markerfacecolor="white",
        capsize=5,
        elinewidth=1.5,
        zorder=5,
        label=r"SCALE-CERT at $M=256$",
    )
    ax.set_xscale("log")
    ax.set_xlabel("retained dimension")
    ax.set_ylabel("population test risk")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=2,
        columnspacing=0.8,
        handlelength=1.5,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.72))
    fig.savefig(FIGURES / "finite_sample_ols.pdf")
    fig.savefig(FIGURES / "finite_sample_ols.png", dpi=180)
    plt.close(fig)


def _synthetic_families() -> dict[str, tuple[float, list[float], list[float]]]:
    return {
        "single_power": (0.20, [0.55], [0.15]),
        "hidden_crossover": (0.20, [0.10, 0.80], [0.00030, 0.15]),
        "three_component": (0.18, [0.12, 0.45, 1.10], [0.00025, 0.020, 0.12]),
    }


def synthetic_benchmark() -> None:
    repetitions = int(os.environ.get("SYNTHETIC_REPETITIONS", "30"))
    bootstrap_repetitions = int(os.environ.get("BOOTSTRAP_REPETITIONS", "40"))
    x = np.geomspace(1.0, 128.0, 9)
    target = 1_000_000.0
    standard_error = 1e-3
    se = np.full(x.size, standard_error)
    delta = 0.10
    broad_grid = np.linspace(0.05, 1.50, 31)
    narrow_grid = np.linspace(0.20, 1.50, 27)
    rng = np.random.default_rng(2027)
    details: list[dict[str, object]] = []

    for family, (floor, exponents, weights) in _synthetic_families().items():
        pilot_truth = mixture_risk(x, floor, exponents, weights, basis="power")
        target_truth = float(mixture_risk([target], floor, exponents, weights, basis="power")[0])
        confidence_threshold = 0.10 * float(np.max(pilot_truth) - np.min(pilot_truth))
        for repetition in range(repetitions):
            observations = pilot_truth + rng.normal(scale=standard_error, size=x.size)
            seed = 10_000 * list(_synthetic_families()).index(family) + repetition
            point, boot_lower, boot_upper, _ = parametric_bootstrap_interval(
                x,
                observations,
                se,
                target,
                model="one_power",
                repetitions=bootstrap_repetitions,
                delta=delta,
                seed=seed,
            )
            lap_point, lap_lower, lap_upper = laplace_floor_power_interval(
                x, observations, se, target, repetitions=1200, delta=delta, seed=seed
            )
            two_fit = fit_two_power(x, observations, se)
            two_point = float(two_fit.predict([target])[0])
            cert_broad = continuous_certificate_interval(
                x, observations, se, target, broad_grid, delta=delta, basis="power"
            )
            cert_narrow = continuous_certificate_interval(
                x, observations, se, target, narrow_grid, delta=delta, basis="power"
            )
            method_values = {
                "one_power_bootstrap": (point, boot_lower, boot_upper),
                "one_power_laplace": (lap_point, lap_lower, lap_upper),
                "continuous_cert_broad": ((cert_broad.lower + cert_broad.upper) / 2.0, cert_broad.lower, cert_broad.upper),
                "continuous_cert_narrow": ((cert_narrow.lower + cert_narrow.upper) / 2.0, cert_narrow.lower, cert_narrow.upper),
                "two_power_point": (two_point, float("nan"), float("nan")),
            }
            for method, (estimate, lower, upper) in method_values.items():
                finite_interval = np.isfinite(lower) and np.isfinite(upper)
                covered = int(finite_interval and lower <= target_truth <= upper)
                width = float(upper - lower) if finite_interval else float("nan")
                details.append(
                    {
                        "family": family,
                        "repetition": repetition,
                        "method": method,
                        "target_truth": target_truth,
                        "estimate": estimate,
                        "absolute_error": abs(estimate - target_truth),
                        "lower": lower,
                        "upper": upper,
                        "width": width,
                        "covered": covered if finite_interval else "",
                        "false_confident": int(finite_interval and not covered and width <= confidence_threshold),
                        "confidence_width_threshold": confidence_threshold,
                    }
                )
    write_csv(RESULTS / "synthetic_benchmark_detail.csv", details)

    summaries: list[dict[str, object]] = []
    for family in _synthetic_families():
        for method in sorted({str(row["method"]) for row in details}):
            subset = [row for row in details if row["family"] == family and row["method"] == method]
            errors = np.asarray([float(row["absolute_error"]) for row in subset])
            finite = [row for row in subset if row["covered"] != ""]
            widths = np.asarray([float(row["width"]) for row in finite]) if finite else np.asarray([])
            summaries.append(
                {
                    "family": family,
                    "method": method,
                    "repetitions": len(subset),
                    "median_absolute_error": float(np.median(errors)),
                    "mean_absolute_error": float(np.mean(errors)),
                    "interval_runs": len(finite),
                    "coverage": float(np.mean([int(row["covered"]) for row in finite])) if finite else "",
                    "median_width": float(np.median(widths)) if widths.size else "",
                    "false_confident_rate": float(np.mean([int(row["false_confident"]) for row in finite])) if finite else "",
                }
            )
    write_csv(RESULTS / "synthetic_benchmark_summary.csv", summaries)

    plotted = [row for row in summaries if row["coverage"] != ""]
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    markers = {"single_power": "o", "hidden_crossover": "s", "three_component": "^"}
    family_labels = {
        "single_power": "one power",
        "hidden_crossover": "hidden crossover",
        "three_component": "three components",
    }
    method_labels = {
        "continuous_cert_broad": "broad certificate",
        "continuous_cert_narrow": "narrow certificate",
        "one_power_bootstrap": "bootstrap",
        "one_power_laplace": "Laplace",
    }
    method_colors = {
        method: f"C{index}"
        for index, method in enumerate(method_labels)
    }
    for row in plotted:
        family = str(row["family"])
        method = str(row["method"])
        ax.scatter(
            float(row["median_width"]),
            float(row["coverage"]),
            marker=markers[family],
            color=method_colors[method],
        )
    ax.axhline(0.90, color="0.35", linestyle=":")
    ax.set_xscale("log")
    ax.set_xlabel("median interval width")
    ax.set_ylabel("target coverage")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("Coverage--informativeness tradeoff")
    legend_handles = [
        Line2D(
            [],
            [],
            color="black",
            marker=markers[family],
            linestyle="None",
            label=label,
        )
        for family, label in family_labels.items()
    ]
    legend_handles.extend(
        Line2D(
            [],
            [],
            color=color,
            marker="o",
            linestyle="None",
            label=method_labels[method],
        )
        for method, color in method_colors.items()
    )
    legend_handles.append(
        Line2D([], [], color="0.35", linestyle=":", label="nominal 90%")
    )
    ax.legend(handles=legend_handles, fontsize=11, ncol=2, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGURES / "synthetic_coverage_width.pdf")
    fig.savefig(FIGURES / "synthetic_coverage_width.png", dpi=180)
    plt.close(fig)


def fetch_public_data() -> None:
    for filename, url in PUBLIC_URLS.items():
        path = PUBLIC / filename
        if not path.exists():
            print(f"Downloading pinned public curve file: {url}")
            urllib.request.urlretrieve(url, path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != PUBLIC_SHA256[filename]:
            raise RuntimeError(
                f"checksum mismatch for {filename}: {digest} != {PUBLIC_SHA256[filename]}"
            )


def _read_public_curves() -> list[tuple[str, str, np.ndarray, np.ndarray]]:
    curves: list[tuple[str, str, np.ndarray, np.ndarray]] = []
    selections = {
        "benchmark.lang.csv": ("NMT", None, None),
        "benchmark.vision.csv": ("IC", "MiX/B/16", {"bird_5", "c_5", "cal_5", "inet_5"}),
    }
    for filename, (domain_keep, model_keep, tasks_keep) in selections.items():
        grouped: dict[tuple[str, str, str], list[tuple[float, float]]] = defaultdict(list)
        with (PUBLIC / filename).open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["Domain"] != domain_keep:
                    continue
                if model_keep is not None and row["Model"] != model_keep:
                    continue
                if tasks_keep is not None and row["Task"] not in tasks_keep:
                    continue
                grouped[(row["Domain"], row["Task"], row["Model"])].append(
                    (float(row["Seen Examples"]), float(row["Loss"]))
                )
        for (domain, task, model), pairs in grouped.items():
            pairs = sorted(set(pairs))
            x = np.asarray([pair[0] for pair in pairs])
            y = np.asarray([pair[1] for pair in pairs])
            if x.size > 24:
                indices = np.unique(np.rint(np.linspace(0, x.size - 1, 24)).astype(int))
                x, y = x[indices], y[indices]
            if x.size >= 8:
                curves.append((domain, f"{task}|{model}", x, y))
    return curves


def public_benchmark() -> None:
    fetch_public_data()
    curves = _read_public_curves()
    grid = np.linspace(0.05, 2.0, 40)
    detail: list[dict[str, object]] = []
    for domain, name, x_full, y_full in curves:
        for fraction in [0.50, 0.70, 0.85]:
            n_pilot = max(5, min(int(round(fraction * x_full.size)), x_full.size - 1))
            x = x_full[:n_pilot]
            y = y_full[:n_pilot]
            target_x = float(x_full[-1])
            target_y = float(y_full[-1])
            x_norm = x / x[0]
            target_norm = target_x / x[0]
            one = fit_floor_power(x_norm, y)
            two = fit_two_power(x_norm, y)
            one_prediction = float(one.predict([target_norm])[0])
            two_prediction = float(two.predict([target_norm])[0])
            residual_scale = max(float(np.sqrt(np.mean((one.predict(x_norm) - y) ** 2))), 1e-6)
            _, boot_lower, boot_upper, _ = parametric_bootstrap_interval(
                x_norm,
                y,
                np.full(y.size, residual_scale),
                target_norm,
                repetitions=50,
                delta=0.10,
                seed=stable_seed(domain, name, n_pilot),
            )
            minimum_radius = minimum_uniform_band_radius(x_norm, y, grid, basis="power")
            for multiplier in [1.0, 2.0, 3.0, 4.0]:
                radius = multiplier * minimum_radius + 1e-9
                certificate = continuous_certificate_interval_from_bands(
                    x_norm,
                    y - radius,
                    y + radius,
                    target_norm,
                    grid,
                    basis="power",
                )
                detail.append(
                    {
                        "domain": domain,
                        "curve": name,
                        "pilot_fraction": fraction,
                        "pilot_points": n_pilot,
                        "target_horizon": target_x / x[-1],
                        "method": f"continuous_cert_{multiplier:.1f}x",
                        "prediction": (certificate.lower + certificate.upper) / 2.0,
                        "absolute_error": abs((certificate.lower + certificate.upper) / 2.0 - target_y),
                        "lower": certificate.lower,
                        "upper": certificate.upper,
                        "width": certificate.width,
                        "covered": int(certificate.lower <= target_y <= certificate.upper),
                        "minimum_pilot_radius": minimum_radius,
                    }
                )
            for method, prediction, lower, upper in [
                ("one_power", one_prediction, float("nan"), float("nan")),
                ("two_power", two_prediction, float("nan"), float("nan")),
                ("one_power_bootstrap", one_prediction, boot_lower, boot_upper),
            ]:
                finite = np.isfinite(lower) and np.isfinite(upper)
                detail.append(
                    {
                        "domain": domain,
                        "curve": name,
                        "pilot_fraction": fraction,
                        "pilot_points": n_pilot,
                        "target_horizon": target_x / x[-1],
                        "method": method,
                        "prediction": prediction,
                        "absolute_error": abs(prediction - target_y),
                        "lower": lower,
                        "upper": upper,
                        "width": upper - lower if finite else "",
                        "covered": int(lower <= target_y <= upper) if finite else "",
                        "minimum_pilot_radius": minimum_radius,
                    }
                )
    write_csv(RESULTS / "public_rolling_origin_detail.csv", detail)

    summary_rows: list[dict[str, object]] = []
    for domain in sorted({str(row["domain"]) for row in detail}):
        for method in sorted({str(row["method"]) for row in detail}):
            subset = [row for row in detail if row["domain"] == domain and row["method"] == method]
            if not subset:
                continue
            interval_rows = [row for row in subset if row["covered"] != ""]
            summary_rows.append(
                {
                    "domain": domain,
                    "method": method,
                    "evaluations": len(subset),
                    "median_absolute_error": float(np.median([float(row["absolute_error"]) for row in subset])),
                    "coverage": float(np.mean([int(row["covered"]) for row in interval_rows])) if interval_rows else "",
                    "median_width": float(np.median([float(row["width"]) for row in interval_rows])) if interval_rows else "",
                }
            )
    write_csv(RESULTS / "public_rolling_origin_summary.csv", summary_rows)

    plot_rows = [
        row for row in summary_rows
        if str(row["method"]).startswith("continuous_cert")
    ]
    fig, ax = plt.subplots(figsize=(6.5, 3.9))
    marker = {"IC": "o", "NMT": "s"}
    for row in plot_rows:
        ax.scatter(
            float(row["median_width"]),
            float(row["coverage"]),
            marker=marker[str(row["domain"])],
            label=f"{row['domain']} / {str(row['method']).split('_')[-1]}",
        )
    ax.set_xscale("log")
    ax.set_xlabel("median interval width")
    ax.set_ylabel("descriptive held-out coverage")
    ax.set_ylim(-0.02, 1.05)
    ax.axhline(0.90, linestyle=":", label="90% reference")
    ax.set_title("Public curves: mismatch sensitivity frontier")
    ax.legend(fontsize=11, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES / "public_coverage_width.pdf")
    fig.savefig(FIGURES / "public_coverage_width.png", dpi=180)
    plt.close(fig)


def run_stage(stage: str) -> None:
    functions = {
        "continuous": continuous_refinement,
        "ols": finite_sample_ols,
        "synthetic": synthetic_benchmark,
        "public": public_benchmark,
    }
    functions[stage]()
    print(f"Completed stage: {stage}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["all", "continuous", "ols", "synthetic", "public"], default="all")
    args = parser.parse_args()
    if args.stage == "all":
        for stage in ["continuous", "ols", "synthetic", "public"]:
            subprocess.run([sys.executable, __file__, "--stage", stage], check=True)
    else:
        run_stage(args.stage)


if __name__ == "__main__":
    main()
