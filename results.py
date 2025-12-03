# results.py 

from __future__ import annotations # Allows type hints that reference classes before they are defined.

import os
import datetime
from dataclasses import dataclass # Provides an easy way to create lightweight data-holding classes.
from typing import Dict, List # Enables clear type hints for lists, dictionaries, and tuples.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from probability_model import (
    Point,
    Click,
    Features,
    MODEL_CLASSES,
    MODEL_PARAMS,
    compute_features,
    compute_posterior,
    rayleigh_hit_probability,
    compute_click_counts,
)


# ---------------------------------------------------------------------
# Exceptions / containers
# ---------------------------------------------------------------------

class NotEnoughDataError(Exception):
    """Raised when the user hasn't drawn/clicked enough to analyze."""


@dataclass
class AnalysisResult:
    summary_text: str
    posterior: Dict[str, float]
    click_stats: List[Dict]
    features: Features
    output_dir: str


# ---------------------------------------------------------------------
# Data sufficiency checks
# ---------------------------------------------------------------------

def _check_minimum_data(
    data_store: Dict[str, List],
    click_radii: List[float],
    min_strokes_per_task: int = 5,
    min_total_clicks: int = 65,
    min_clicks_per_button: int = 8,
) -> None:
    """Raise NotEnoughDataError if inputs are too small."""
    # Strokes
    for task in ("line", "circle", "spiral", "eight"):
        n = len(data_store.get(task, []))
        if n < min_strokes_per_task:
            raise NotEnoughDataError(
                f"Please draw at least {min_strokes_per_task} strokes for each task.\n"
                f"Currently: {task} has only {n} stroke(s)."
            )

    # Clicks (total)
    clicks: List[Click] = data_store.get("clicks", [])
    total_clicks = len(clicks)
    if total_clicks < min_total_clicks:
        raise NotEnoughDataError(
            f"Please make at least {min_total_clicks} clicks total.\n"
            f"Currently you have {total_clicks}."
        )

    # Clicks per button size
    counts = compute_click_counts(clicks, click_radii)
    for i, r in enumerate(click_radii):
        if counts[i]["total"] < min_clicks_per_button:
            diam = 2 * r
            raise NotEnoughDataError(
                f"Please make at least {min_clicks_per_button} clicks on each button size.\n"
                f"The {diam}px buttons currently have only {counts[i]['total']} click(s)."
            )


# ---------------------------------------------------------------------
# Text summary
# ---------------------------------------------------------------------

def _build_text_summary(
    features: Features,
    posterior: Dict[str, float],
    click_radii: List[float],
    click_stats: List[Dict],
) -> str:
    best_class = max(posterior, key=posterior.get)
    best_prob = posterior[best_class] 

    # Comfort interpretation
    if best_class == "robot-perfect":
        comfort = (
            "Big, medium, and even quite small buttons should feel very easy "
            "for you under this toy model."
        )
    elif best_class == "typical / healthy":
        comfort = (
            "Big and medium buttons should feel comfortable. Very tiny buttons "
            "may still be annoying."
        )
    elif best_class == "mild tremor-like":
        comfort = (
            "Big buttons are comfortable, medium ones may sometimes be missed, "
            "and tiny ones can be frustrating."
        )
    else:  # impaired access
        comfort = (
            "This pattern suggests that small targets may be genuinely hard to hit. "
            "Interfaces that rely on tiny buttons may be unfair to hands like yours."
        )

    lines = []
    lines.append("🌟 Quick summary")
    lines.append(f"Most similar synthetic profile: {best_class} ({best_prob})")
    lines.append(f"Button comfort: {comfort}")
    lines.append("")

    # Button stats
    sigma = features.click_sd
    lines.append("🎯 Button sizes – model vs your data")
    for i, r in enumerate(click_radii):
        diam = 2 * r
        predicted = click_stats[i]["predicted"]
        hits = click_stats[i]["hits"]
        total = click_stats[i]["total"]
        empirical = click_stats[i]["empirical"]

        if predicted is None:
            pred_str = "model P(hit): not enough data"
        else:
            pred_str = f"model P(hit) ≈ {predicted}"

        if total > 0 and empirical is not None:
            emp_str = f"your data: {hits}/{total} hits ({empirical})"
        else:
            emp_str = "your data: (no clicks)"

        lines.append(f"  {diam:>3.0f}px buttons: {pred_str} | {emp_str}")
    lines.append("")

    # Movement features
    lines.append("🔬 Movement features (estimates to 3 decimals)")
    lines.append(f"  Line deviation SD (px):            {features.line_jitter_sd:.3f}")
    lines.append(f"  Line zero-crossing freq (/sec):    {features.crossing_freq:.3f}")
    lines.append(f"  Angle jitter SD (rad):             {features.angle_jitter_sd:.3f}")
    lines.append(f"  Step size variance:                {features.step_var:.3f}")
    lines.append(f"  Radial deviation SD (px):          {features.radial_sd:.3f}")
    lines.append(f"  Click error SD (px):               {features.click_sd:.3f}")
    lines.append("")

    # Posterior over profiles
    lines.append("Posterior over synthetic motor profiles (Gaussian Naive Bayes):")
    for cls in MODEL_CLASSES:
        lines.append(f"  {cls}: {posterior[cls]}")
    lines.append("")

    # Interpretation
    lines.append("Interpretation notes:")
    lines.append(
        "- Lower variability (especially line deviation and jitter) looks more "
        "robot-like. Higher variability is very human and can reflect many things:"
    )
    lines.append(
        "  mouse vs trackpad, stress, tiredness, caffeine, cold hands, tremor, "
        "and other motor conditions."
    )
    lines.append(
        "- These categories are synthetic examples, not diagnoses yet. The point is to "
        "see how random variables and Bayes' rule can describe motor noise."
    )
    lines.append("")

    # Accessibility reflection via Rayleigh
    if sigma is not None and sigma > 0:
        lines.append("♿ Accessibility reflection")
        lines.append(
            f"Based on your estimated click error noise (SD ≈ {sigma:.3f} px), "
            "the Rayleigh model suggests roughly:"
        )
        for r in click_radii:
            diam = 2 * r
            p_hit = rayleigh_hit_probability(r, sigma)
            if p_hit is not None:
                lines.append(f"  • ~{diam:>3.0f}px buttons: success (P(hit)) ≈ {p_hit}")
        lines.append("")
        lines.append(
            "Interfaces that rely heavily on tiny click targets implicitly expect low "
            "motor noise. Users with higher variability (including tremor, disabilities, "
            "or just a finicky touchpad) will have a systematically worse experience."
        )
    else:
        lines.append("Not enough click data to estimate pointing noise yet.")

    return "\n".join(lines)


# ---------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------

def _plot_posterior_bar(
    posterior: Dict[str, float],
    output_dir: str,
    filename: str = "posterior_profiles.png",
) -> None:
    probs = [posterior[cls] for cls in MODEL_CLASSES]
    labels = MODEL_CLASSES

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(labels, probs)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Probability")
    fig.text(
        0.5,          # x position (centered)
        0.001,         # y position (very bottom of the figure)
        "Posterior over synthetic motor profiles",
        ha="center",
        va="bottom",
        fontsize=12
    )
    for i, p in enumerate(probs):
        ax.text(i, p + 0.02, f"{p}", ha="center", va="bottom", fontsize=7)

    plt.subplots_adjust(bottom=0.80)
    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _plot_click_noise_gaussians(
    features: Features,
    output_dir: str,
    filename: str = "click_error_gaussians.png",
) -> None:
    """
    Plot Gaussian pdfs for the click-error SD feature and mark the user's SD.
    (Fixes the ambiguous max() on NumPy arrays.)
    """
    f_idx = 5  # click_sd index
    mus = np.array([MODEL_PARAMS[cls]["mu"][f_idx] for cls in MODEL_CLASSES])
    sds = np.array([MODEL_PARAMS[cls]["sd"][f_idx] for cls in MODEL_CLASSES])

    # upper bound from Gaussians
    upper_gauss = float(np.max(mus + 3 * sds))
    # upper bound from the user's SD
    upper_click = features.click_sd * 1.2 if features.click_sd > 0 else 0.0
    x_max = max(upper_gauss, upper_click, 1.0)

    x = np.linspace(0, x_max, 300)
    fig, ax = plt.subplots(figsize=(6, 3))

    colors = ["C0", "C1", "C2", "C3"]
    for idx, cls in enumerate(MODEL_CLASSES):
        mu = MODEL_PARAMS[cls]["mu"][f_idx]
        sd = MODEL_PARAMS[cls]["sd"][f_idx]
        pdf = (1.0 / (sd * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sd) ** 2)
        ax.plot(x, pdf, color=colors[idx], label=cls)

    if features.click_sd > 0:
        ax.axvline(features.click_sd, color="k", linestyle="--", linewidth=1)
        ax.text(
            features.click_sd,
            ax.get_ylim()[1] * 0.9,
            f"you: {features.click_sd:.3f}px",
            rotation=90,
            va="top",
            ha="right",
            fontsize=8,
        )

    ax.set_xlabel("Click error SD (px)")
    ax.set_ylabel("Gaussian pdf")
    ax.set_title("How each profile's click-error-noise model sees you")
    ax.legend(fontsize=8)
    plt.tight_layout()

    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _plot_rayleigh_hit_probabilities(
    click_radii: List[float],
    click_stats: List[Dict],
    features: Features,
    output_dir: str,
    filename: str = "rayleigh_hit_probabilities.png",
) -> None:
    """
    Rayleigh CDF vs radius, with markers at the button radii.
    """
    sigma = features.click_sd
    if sigma is None or sigma <= 0:
        return

    max_r = max(click_radii)
    r = np.linspace(0, max_r * 1.2, 300)
    p_hit = 1.0 - np.exp(-(r ** 2) / (2.0 * sigma ** 2))

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(r, p_hit, label="Rayleigh model P(hit)")

    for stat in click_stats:
        rad = stat["radius"]
        model_p = stat["predicted"]
        ax.scatter(rad, model_p, color="C1", zorder=5)

    ax.set_xlabel("Button radius (px)")
    ax.set_ylabel("P(hit)")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Rayleigh hit probability vs button radius")
    ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _plot_click_error_histogram(
    clicks: List[Click],
    output_dir: str,
    filename: str = "click_error_histogram.png",
) -> None:
    """
    Histogram of raw click radial errors.
    """
    errors = [c.error_px for c in clicks]
    if not errors:
        return

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(errors, bins=30, edgecolor="black")
    ax.set_xlabel("Click radial error (px)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of click errors")
    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _plot_button_accuracy(
    click_radii: List[float],
    click_stats: List[Dict],
    output_dir: str,
    filename: str = "button_accuracy_empirical_vs_model.png",
) -> None:
    """
    Grouped bar chart: model vs empirical hit probability per button diameter.
    """
    labels = [f"{2 * r:.0f}px" for r in click_radii]
    predicted = [s["predicted"] if s["predicted"] is not None else 0.0 for s in click_stats]
    empirical = [s["empirical"] if s["empirical"] is not None else 0.0 for s in click_stats]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(x - width / 2, predicted, width, label="Model", alpha=0.85)
    ax.bar(x + width / 2, empirical, width, label="Your data", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Hit probability")
    ax.set_title("Button hit probability: model vs empirical")
    ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------
# Saving data
# ---------------------------------------------------------------------

def _save_raw_data_txt(
    data_store: Dict[str, List],
    click_radii: List[float],
    output_dir: str,
    filename: str = "raw_data.txt",
) -> None:
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Raw data from Little Tremors, Big Stories\n\n")

        # Strokes
        for task in ("line", "circle", "spiral", "eight"):
            strokes = data_store.get(task, [])
            f.write(f"[Strokes – {task}] count={len(strokes)}\n")
            for s_id, stroke in enumerate(strokes):
                f.write(f"  stroke {s_id} (n={len(stroke)}):\n")
                for i, p in enumerate(stroke):
                    f.write(
                        f"    {i:4d}: x={p.x:.3f}, y={p.y:.3f}, t_ms={p.t_ms:.1f}\n"
                    )
            f.write("\n")

        # Clicks
        clicks: List[Click] = data_store.get("clicks", [])
        f.write(f"[Clicks] count={len(clicks)}\n")
        for i, c in enumerate(clicks):
            diam = (
                2 * click_radii[c.target_index]
                if 0 <= c.target_index < len(click_radii)
                else None
            )
            f.write(
                f"  {i:4d}: x={c.x:.3f}, y={c.y:.3f}, t_ms={c.t_ms:.1f}, "
                f"target_index={c.target_index}, target_diam={diam}, "
                f"hit={int(c.hit)}, error_px={c.error_px:.3f}\n"
            )


def _save_summary_txt(
    summary_text: str,
    output_dir: str,
    filename: str = "summary.txt",
) -> None:
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(summary_text)


# ---------------------------------------------------------------------
# Top-level API
# ---------------------------------------------------------------------

def analyze_and_save(
    data_store: Dict[str, List],
    canvas_width: float,
    canvas_height: float,
    click_radii: List[float],
    base_output_dir: str = "Results",
) -> AnalysisResult:
    """
    Main entry point for the UI:
      - checks data size
      - computes features and posterior
      - builds text summary
      - saves raw data, summary, and multiple graphs into a new folder
    """
    _check_minimum_data(data_store, click_radii)

    features = compute_features(data_store, canvas_width, canvas_height)
    posterior = compute_posterior(features)

    clicks: List[Click] = data_store.get("clicks", [])
    counts = compute_click_counts(clicks, click_radii)

    click_stats: List[Dict] = []
    for i, r in enumerate(click_radii):
        hits = counts[i]["hits"]
        total = counts[i]["total"]
        empirical = (hits / total) if total > 0 else None
        predicted = rayleigh_hit_probability(r, features.click_sd)
        click_stats.append(
            {
                "radius": r,
                "hits": hits,
                "total": total,
                "empirical": empirical,
                "predicted": predicted,
            }
        )

    summary_text = _build_text_summary(features, posterior, click_radii, click_stats)

    ts = datetime.datetime.now().strftime("%b-%d-%Y-%I-%M-%S%p")
    output_dir = os.path.join(base_output_dir, f"session-{ts}")
    os.makedirs(output_dir, exist_ok=True)

    _save_raw_data_txt(data_store, click_radii, output_dir)
    _save_summary_txt(summary_text, output_dir)
    _plot_posterior_bar(posterior, output_dir)
    _plot_click_noise_gaussians(features, output_dir)
    _plot_rayleigh_hit_probabilities(click_radii, click_stats, features, output_dir)
    _plot_click_error_histogram(clicks, output_dir)
    _plot_button_accuracy(click_radii, click_stats, output_dir)

    return AnalysisResult(
        summary_text=summary_text,
        posterior=posterior,
        click_stats=click_stats,
        features=features,
        output_dir=output_dir,
    )
