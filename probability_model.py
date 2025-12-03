# probability_model.py (ChatGPT gave some ideas on implementation, but most of the code here is original)

from __future__ import annotations # Allows type hints that reference classes before they are defined.

from dataclasses import dataclass # Provides an easy way to create lightweight data-holding classes.
from typing import List, Dict, Tuple # Enables clear type hints for lists, dictionaries, and tuples.

import math
import numpy as np


# ---------------------------------------------------------------------
# Basic data structures
# ---------------------------------------------------------------------

@dataclass
class Point:
    x: float
    y: float
    t_ms: float  # time in milliseconds


@dataclass
class Click:
    x: float
    y: float
    t_ms: float
    target_index: int
    hit: bool
    error_px: float


@dataclass
class Features:
    line_jitter_sd: float
    crossing_freq: float
    angle_jitter_sd: float
    step_var: float
    radial_sd: float
    click_sd: float


# ---------------------------------------------------------------------
# Helper stats
# ---------------------------------------------------------------------

def mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def std_dev(values: List[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    m = mean(values)
    s2 = sum((v - m) ** 2 for v in values) / (n - 1) # from sample variance
    return float(math.sqrt(s2))


def normal_log_pdf(x: float, mu: float, sigma: float) -> float:
    """Log of N(x; mu, sigma^2)."""
    s = sigma if sigma > 0 else 1.0
    z = (x - mu) / s
    return -math.log(math.sqrt(2 * math.pi) * s) - 0.5 * z * z


# ---------------------------------------------------------------------
# Feature extraction from strokes / clicks
# ---------------------------------------------------------------------

def stroke_kinematics(points: List[Point]) -> Dict[str, List[float]]:
    """
    Given a stroke (list of Points), compute step sizes, angles,
    angle differences, and time deltas.
    """
    if len(points) < 2:
        return {"steps": [], "angles": [], "angle_diffs": [], "dt": []}

    steps: List[float] = []
    angles: List[float] = []
    dt: List[float] = []

    for i in range(1, len(points)):
        dx = points[i].x - points[i - 1].x
        dy = points[i].y - points[i - 1].y

        dist = math.sqrt(dx * dx + dy * dy) # step size 
        angle = math.atan2(dy, dx)
        delta_t = points[i].t_ms - points[i - 1].t_ms

        steps.append(dist)
        angles.append(angle)
        dt.append(delta_t)

    angle_diffs: List[float] = []
    for i in range(1, len(angles)):
        diff = angles[i] - angles[i - 1]
        # wrap to [-pi, pi]
        while diff > math.pi:
            diff -= 2 * math.pi
        while diff < -math.pi:
            diff += 2 * math.pi
        angle_diffs.append(diff)

    return {
        "steps": steps,
        "angles": angles,
        "angle_diffs": angle_diffs,
        "dt": dt,
    }


def line_deviations(
    line_strokes: List[List[Point]],
    canvas_height: float
) -> Tuple[List[float], int, float]:
    """
    Measure vertical deviations from the horizontal midline and
    count zero-crossings over time.

    Returns:
        (devs, zero_crossings_count, total_time_sec)
    """
    devs: List[float] = []
    zero_crossings = 0
    y0 = canvas_height / 2.0 # center line

    last_sign = None
    first_t: float | None = None # here | is a union type
    last_t: float | None = None

    for stroke in line_strokes:
        for p in stroke:
            d = p.y - y0 # deviation from midline
            devs.append(d)

            sign = 0
            if d > 0:
                sign = 1
            elif d < 0:
                sign = -1

            if first_t is None:
                first_t = p.t_ms
            last_t = p.t_ms

            if last_sign is not None and sign != 0 and sign != last_sign:
                zero_crossings += 1 # crossed midline
            if sign != 0:
                last_sign = sign

    total_time_sec = 0.0
    if first_t is not None and last_t is not None and last_t > first_t:
        total_time_sec = (last_t - first_t) / 1000.0 # time per test in seconds

    return devs, zero_crossings, total_time_sec


def radial_deviations(
    circle_strokes: List[List[Point]],
    spiral_strokes: List[List[Point]],
    eight_strokes: List[List[Point]],
    canvas_width: float,
    canvas_height: float
) -> List[float]:
    """
    Compute deviation from average radius for all points in circle/spiral/8 strokes.
    Spiral and figure-8 strokes are included together with circles to provide
    additional curved-motion samples. We treat them as contributors to an overall
    “radial spread” feature (how far radii vary from the center), rather than
    evaluating spiral/8 accuracy separately.
    """
    cx = canvas_width / 2.0
    cy = canvas_height / 2.0

    all_points: List[Point] = []
    for s in circle_strokes + spiral_strokes + eight_strokes:
        all_points.extend(s)

    if not all_points:
        return []

    dists = [
        math.sqrt((p.x - cx) ** 2 + (p.y - cy) ** 2)
        for p in all_points
    ]
    r_mean = mean(dists)
    radial_devs = [d - r_mean for d in dists]
    return radial_devs


def compute_click_sd(clicks: List[Click]) -> float:
    """
    Standard deviation of radial error in clicks.
    """
    errors = [c.error_px for c in clicks]
    return std_dev(errors)


def compute_click_counts(
    clicks: List[Click],
    click_radii: List[float]
) -> List[Dict[str, int]]:
    """
    Count hits/total per button index.
    """
    counts = [{"hits": 0, "total": 0} for _ in click_radii]
    for c in clicks:
        if 0 <= c.target_index < len(click_radii):
            counts[c.target_index]["total"] += 1
            if c.hit:
                counts[c.target_index]["hits"] += 1
    return counts


def compute_features(
    data_store: Dict[str, List],
    canvas_width: float,
    canvas_height: float
) -> Features:
    """
    Convert raw strokes + clicks into the 6 scalar features used in the model.
    data_store keys:
        'line', 'circle', 'spiral', 'eight' -> list of list[Point]
        'clicks' -> list[Click]
    """
    # 1 & 2) Linear Standard Deviation & Zero Crossing Frequency:
    devs, zero_count, total_time = line_deviations(
        data_store.get("line", []),
        canvas_height
    )
    line_jitter_sd = std_dev(devs)
    crossing_freq = (zero_count / total_time) if total_time > 0 else 0.0

    # 3 & 4) Angle Jitter Standard Deviation & Step Size Variance:
    all_angle_diffs: List[float] = []
    all_steps: List[float] = []

    for task in ("line", "circle", "spiral", "eight"):
        for stroke in data_store.get(task, []):
            kin = stroke_kinematics(stroke)
            all_angle_diffs.extend(kin["angle_diffs"])
            all_steps.extend(kin["steps"])

    angle_jitter_sd = std_dev(all_angle_diffs)
    step_sd = std_dev(all_steps)
    step_var = step_sd ** 2

    # 5) Radial Standard Deviation for circle/spiral/8:
    radial_devs = radial_deviations(
        data_store.get("circle", []),
        data_store.get("spiral", []),
        data_store.get("eight", []),
        canvas_width,
        canvas_height
    )
    radial_sd = std_dev(radial_devs)

    # 6) Click Error Standard Deviation:
    click_sd = compute_click_sd(data_store.get("clicks", []))

    return Features(
        line_jitter_sd=line_jitter_sd,
        crossing_freq=crossing_freq,
        angle_jitter_sd=angle_jitter_sd,
        step_var=step_var,
        radial_sd=radial_sd,
        click_sd=click_sd,
    )


# ---------------------------------------------------------------------
# Probability model parameters and inference
# ---------------------------------------------------------------------

MODEL_CLASSES = [
    "robot-perfect",
    "typical / healthy",
    "mild tremor-like",
    "impaired access",
]

# order of features:
# [line_jitter_sd, crossing_freq, angle_jitter_sd, step_var, radial_sd, click_sd]
MODEL_PARAMS: Dict[str, Dict[str, List[float]]] = {
    "robot-perfect": {
        "mu": [1.0, 0.5, 0.20, 1.5, 10.0, 1.5],
        "sd": [0.8, 0.5, 0.10, 1.0, 5.0, 0.8],
    },
    "typical / healthy": {
        "mu": [3.0, 1.5, 0.60, 4.0, 25.0, 3.0],
        "sd": [1.5, 0.8, 0.25, 2.5, 10.0, 1.5],
    },
    "mild tremor-like": {
        "mu": [6.0, 2.5, 0.90, 7.0, 40.0, 6.0],
        "sd": [2.0, 1.0, 0.30, 3.0, 15.0, 2.5],
    },
    "impaired access": {
        "mu": [9.0, 4.0, 1.30, 12.0, 60.0, 10.0],
        "sd": [3.0, 1.5, 0.50, 5.0, 20.0, 4.0],
    },
}

MODEL_PRIORS: Dict[str, float] = {
    "robot-perfect": 0.05,
    "typical / healthy": 0.60,
    "mild tremor-like": 0.25,
    "impaired access": 0.10,
}


def compute_posterior(features: Features) -> Dict[str, float]:
    """
    Gaussian Naive Bayes posterior over MODEL_CLASSES.
    """
    x = [
        features.line_jitter_sd,
        features.crossing_freq,
        features.angle_jitter_sd,
        features.step_var,
        features.radial_sd,
        features.click_sd,
    ]

    log_post: Dict[str, float] = {}

    # Initialize to negative infinity so that any real log-probability
    # will be larger and become the new maximum.
    max_log = -float("inf")

    for cls in MODEL_CLASSES:
        mu = MODEL_PARAMS[cls]["mu"]
        sd = MODEL_PARAMS[cls]["sd"]
        lp = math.log(MODEL_PRIORS[cls])

        for j in range(len(x)):
            feature_value = x[j]
            mean = mu[j]
            std = sd[j]
            lp += normal_log_pdf(feature_value, mean, std)

        log_post[cls] = lp

        # We subtract the largest log-likelihood so exponentials stay in a safe numeric range.
        # This prevents underflow when log-probabilities are very negative (log-sum-exp trick).
        # Suggested by ChatGPT
        if lp > max_log:
            max_log = lp

    # Normalize
    probs: Dict[str, float] = {}
    denom = sum(math.exp(lp - max_log) for lp in log_post.values())
    for cls in MODEL_CLASSES:
        probs[cls] = math.exp(log_post[cls] - max_log) / denom

    return probs


# ---------------------------------------------------------------------
# Rayleigh click model
# ---------------------------------------------------------------------

def rayleigh_hit_probability(radius: float, sigma: float) -> float | None:
    """
    P(hit) for a circular button of radius R, assuming Rayleigh-distributed
    radial error with parameter sigma.
    """
    if sigma is None or sigma <= 0:
        return None
    ratio = (radius ** 2) / (2.0 * sigma ** 2)
    return 1.0 - math.exp(-ratio)
