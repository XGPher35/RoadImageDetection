import argparse
import glob
import json
import os
import sys

import cv2
import pandas as pd

# Reuse the team's single source of truth for SI math / constants.
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from severity.si_utils import (
    CLASS_NAMES,
    CLASS_WEIGHTS,
    grade_severity,
    temporal_smooth_ewma,
)

INFER_IMGSZ = 1280        # train was 640; up-scale at inference for thin cracks
INFER_TTA = True          # test-time augmentation (Ultralytics `augment=True`)
INFER_IOU = 0.6           # NMS IoU
GLOBAL_CONF_FLOOR = 0.20  # low floor; per-class thresholds applied in post

PER_CLASS_CONF = {
    0: 0.20,   # Longitudinal crack  (recover recall)
    1: 0.20,   # Transverse crack    (recover recall)
    2: 0.30,   # Alligator crack
    3: 0.40,   # Pothole             (keep precision high)
}

# Map marker colour per grade (matches backend/severity_map.py).
GRADE_COLOR = {"Good": "green", "Fair": "lightgreen", "Poor": "orange", "Critical": "red"}

# Kathmandu-valley centre for synthetic-coordinate fallback.
DEFAULT_CENTER = (27.667, 85.44)


# ── Coordinate handling ───────────────────────────────────────────────────────

def load_coords(coords_path, image_names):
    """Return {image_name: (lat, lon)}.

    Accepts either a CSV with columns image,lat,long or a severity_points-style
    JSON list. 
    """
    mapping = {}

    if coords_path and os.path.exists(coords_path):
        if coords_path.lower().endswith(".csv"):
            df = pd.read_csv(coords_path)
            for _, row in df.iterrows():
                mapping[str(row["image"])] = (float(row["lat"]), float(row["long"]))
        else:  # JSON
            with open(coords_path) as f:
                data = json.load(f)
            for i, entry in enumerate(data):
                loc = entry.get("location", {})
                lat = loc.get("lat", entry.get("lat"))
                lon = loc.get("long", entry.get("long"))
                key = entry.get("image", entry.get("name", str(i)))
                if lat is not None and lon is not None:
                    mapping[str(key)] = (float(lat), float(lon))

    # Fill any gaps with a synthetic route so the map always renders.
    base_lat, base_lon = DEFAULT_CENTER
    coords = {}
    for i, name in enumerate(image_names):
        if name in mapping:
            coords[name] = mapping[name]
        else:
            # ~15 m steps eastward to simulate driving along a road segment.
            coords[name] = (base_lat + i * 0.00018, base_lon + i * 0.00022)
    return coords


# ── Inference + SI (Option A applied) ─────────────────────────────────────────

def score_image(model, img_path):
    """Run inference with Option A settings and return (SI, n_detections)."""
    img = cv2.imread(img_path)
    if img is None:
        return None, 0
    h, w = img.shape[:2]
    frame_area = float(w * h)

    results = model(
        img_path,
        imgsz=INFER_IMGSZ,
        augment=INFER_TTA,
        conf=GLOBAL_CONF_FLOOR,
        iou=INFER_IOU,
        verbose=False,
    )

    si = 0.0
    n = 0
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        for box in r.boxes:
            cls = int(box.cls)
            conf = float(box.conf)
            # Per-class confidence filtering (Option A #2).
            if conf < PER_CLASS_CONF.get(cls, GLOBAL_CONF_FLOOR):
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            rel_area = ((x2 - x1) * (y2 - y1)) / frame_area
            si += CLASS_WEIGHTS.get(cls, 0.1) * conf * rel_area
            n += 1
    return si, n


# ── Folium map ────────────────────────────────────────────────────────────────

def render_map(df, map_out):
    """Render a colour-coded Folium map with a heatmap layer + layer control."""
    try:
        import folium
        from folium.plugins import HeatMap
    except ImportError:
        print("[map] folium not installed — skipping map render (pip install folium).")
        return

    center = [df["lat"].mean(), df["long"].mean()]
    m = folium.Map(location=center, zoom_start=14)

    markers = folium.FeatureGroup(name="All damage")
    critical = folium.FeatureGroup(name="Critical only")
    heat = []

    for _, row in df.iterrows():
        color = GRADE_COLOR.get(row["segment_grade"], "red")
        popup = (
            f"<b>{row['image']}</b><br>"
            f"Rank: {row['rank']}<br>"
            f"SI: {row['segment_aggregated_SI']}<br>"
            f"Grade: {row['segment_grade']}"
        )
        folium.Marker(
            [row["lat"], row["long"]],
            popup=popup,
            icon=folium.Icon(color=color, icon="info-sign"),
        ).add_to(markers)
        if row["segment_grade"] == "Critical":
            folium.Marker(
                [row["lat"], row["long"]], popup=popup,
                icon=folium.Icon(color="red", icon="exclamation-sign"),
            ).add_to(critical)
        heat.append([row["lat"], row["long"], float(row["segment_aggregated_SI"])])

    markers.add_to(m)
    critical.add_to(m)
    folium.FeatureGroup(name="Heatmap").add_child(HeatMap(heat, radius=18)).add_to(m)
    folium.LayerControl().add_to(m)

    os.makedirs(os.path.dirname(os.path.abspath(map_out)), exist_ok=True)
    m.save(map_out)
    print(f"[map] Saved interactive map -> {map_out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Road Damage Detection master pipeline")
    parser.add_argument("--images_dir", required=True, help="Directory of road-segment frames")
    parser.add_argument("--model", default="backend/model/weights/best.pt", help="YOLOv8 weights")
    parser.add_argument("--coords", default=None, help="Optional CSV/JSON of GPS coords per image")
    parser.add_argument("--csv_out", default="ranked_severity.csv", help="Output ranked CSV")
    parser.add_argument("--map_out", default="maps/severity_map.html", help="Output Folium map")
    parser.add_argument("--alpha", type=float, default=0.3, help="EWMA smoothing alpha")
    args = parser.parse_args()

    if not os.path.isdir(args.images_dir):
        sys.exit(f"Error: {args.images_dir} is not a directory.")

    image_paths = sorted(
        glob.glob(os.path.join(args.images_dir, "*.jpg"))
        + glob.glob(os.path.join(args.images_dir, "*.png"))
        + glob.glob(os.path.join(args.images_dir, "*.jpeg"))
    )
    if not image_paths:
        sys.exit(f"No images found in {args.images_dir}")

    print(f"[model] Loading {args.model} ...")
    from ultralytics import YOLO
    model = YOLO(args.model)

    print(f"[infer] {len(image_paths)} frames @ imgsz={INFER_IMGSZ}, TTA={INFER_TTA} ...")
    rows, raw_si = [], []
    for p in image_paths:
        si, n = score_image(model, p)
        if si is None:
            print(f"  ! skipped unreadable image: {p}")
            continue
        raw_si.append(si)
        rows.append({
            "image": os.path.basename(p),
            "detections": n,
            "per_frame_SI": round(si, 5),
            "grade": grade_severity(si),
        })

    # Temporal smoothing -> segment-level score (Adarsha Ph3/Ph4).
    smoothed = temporal_smooth_ewma(raw_si, alpha=args.alpha)
    for i, row in enumerate(rows):
        row["segment_aggregated_SI"] = round(smoothed[i], 5)
        row["segment_grade"] = grade_severity(smoothed[i])

    # Attach Sandesh's coordinates (Adarsha Ph4 handoff).
    coords = load_coords(args.coords, [r["image"] for r in rows])
    for row in rows:
        lat, lon = coords[row["image"]]
        row["lat"], row["long"] = round(lat, 6), round(lon, 6)

    # Rank by aggregated severity (1 = worst) and export.
    df = pd.DataFrame(rows).sort_values("segment_aggregated_SI", ascending=False)
    df.insert(0, "rank", range(1, len(df) + 1))
    df = df[["rank", "image", "lat", "long", "detections",
             "per_frame_SI", "grade", "segment_aggregated_SI", "segment_grade"]]

    os.makedirs(os.path.dirname(os.path.abspath(args.csv_out)) or ".", exist_ok=True)
    df.to_csv(args.csv_out, index=False)
    print(f"[csv]  Ranked severity report -> {args.csv_out}")
    print(df.head(10).to_string(index=False))

    render_map(df, args.map_out)
    print("\n[done] Pipeline complete.")


if __name__ == "__main__":
    main()
