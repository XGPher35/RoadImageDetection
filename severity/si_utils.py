"""
severity/si_utils.py — Shared Severity Index utilities.

Single source of truth for class definitions, severity weights,
grading thresholds, SI calculation, and temporal-smoothing functions.

All scripts (apply_real_si.py, backend/api.py, generate_segment_report.py)
import from this module to avoid constant/logic duplication.
"""

import numpy as np

# ── Class definitions ────────────────────────────────────────────────────────

CLASS_NAMES = {0: "Longitudinal", 1: "Transverse", 2: "Alligator", 3: "Pothole"}
CLASS_WEIGHTS = {0: 0.5, 1: 0.3, 2: 0.8, 3: 1.0}
CLASS_COLORS = {0: "#3498db", 1: "#2ecc71", 2: "#e67e22", 3: "#e74c3c"}

GRADE_THRESHOLDS = [
    (0.005, "Good"),
    (0.02,  "Fair"),
    (0.05,  "Poor"),
    (float("inf"), "Critical"),
]


# ── Grading ──────────────────────────────────────────────────────────────────

def grade_severity(si: float) -> str:
    """Map a raw SI score to a human-readable grade."""
    for threshold, label in GRADE_THRESHOLDS:
        if si < threshold:
            return label
    return "Critical"


# ── Core SI calculation ─────────────────────────────────────────────────────

def calculate_severity_index(results, frame_area, weights=None):
    """Calculate the Severity Index from YOLOv8 Results objects.

    Args:
        results: List of ultralytics Results objects from model().
        frame_area: Total pixel area of the frame (width * height).
        weights: Optional dict mapping class_id -> severity weight.
                 Defaults to CLASS_WEIGHTS.

    Returns:
        Tuple of (severity_index, details_list).
        Each detail dict contains class_id, class_name, confidence,
        bbox_area, relative_area, and weighted_contribution.
    """
    if weights is None:
        weights = CLASS_WEIGHTS

    severity_index = 0.0
    details = []

    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue

        for box in r.boxes:
            class_id = int(box.cls)
            conf = float(box.conf)
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            bbox_area = (x2 - x1) * (y2 - y1)
            relative_area = bbox_area / frame_area

            w = weights.get(class_id, 0.1)
            contribution = w * conf * relative_area
            severity_index += contribution

            details.append({
                "class_id": class_id,
                "class_name": CLASS_NAMES.get(class_id, f"Unknown({class_id})"),
                "confidence": round(conf, 4),
                "bbox": [round(v, 1) for v in [x1, y1, x2, y2]],
                "bbox_area_px": round(bbox_area, 1),
                "relative_area": round(relative_area, 6),
                "weight": w,
                "contribution": round(contribution, 6),
            })

    return severity_index, details


# ── Temporal smoothing ───────────────────────────────────────────────────────

def temporal_smooth_sma(si_scores, window_size=5):
    """Simple Moving Average smoothing for a sequence of SI scores.

    Args:
        si_scores: List/array of per-frame SI values.
        window_size: Number of frames in the sliding window.

    Returns:
        List of smoothed SI values (same length as input; edges
        use available data — i.e. a centered window clipped at
        boundaries).
    """
    scores = np.array(si_scores, dtype=float)
    n = len(scores)
    smoothed = np.zeros(n)
    half = window_size // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        smoothed[i] = scores[lo:hi].mean()
    return smoothed.tolist()


def temporal_smooth_ewma(si_scores, alpha=0.3):
    """Exponentially Weighted Moving Average smoothing.

    Args:
        si_scores: List/array of per-frame SI values.
        alpha: Smoothing factor in (0, 1].  Higher alpha gives more
               weight to recent observations.

    Returns:
        List of EWMA-smoothed SI values (same length as input).
    """
    scores = np.array(si_scores, dtype=float)
    n = len(scores)
    smoothed = np.zeros(n)
    smoothed[0] = scores[0]
    for i in range(1, n):
        smoothed[i] = alpha * scores[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed.tolist()


def aggregate_segment(si_scores, method="ewma", **kwargs):
    """Aggregate per-frame SI scores into a single road-segment score.

    Args:
        si_scores: List of per-frame SI values.
        method: "sma" or "ewma".
        **kwargs: Forwarded to the chosen smoothing function
                  (window_size for SMA, alpha for EWMA).

    Returns:
        Tuple of (aggregated_si, grade_str).
        The aggregated SI is the mean of the smoothed series.
    """
    if len(si_scores) == 0:
        return 0.0, "Good"

    if method == "sma":
        smoothed = temporal_smooth_sma(si_scores, **kwargs)
    else:
        smoothed = temporal_smooth_ewma(si_scores, **kwargs)

    agg = float(np.mean(smoothed))
    return round(agg, 6), grade_severity(agg)
