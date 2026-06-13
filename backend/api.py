"""
FastAPI backend for SHP — Road Damage Detection & Severity Index.
Serves YOLOv8 inference and severity calculation via REST endpoints.
"""

import io
import base64
from pathlib import Path

from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
from severity_map import generate_map

# ── Constants ────────────────────────────────────────────────────────────────
import sys
import os

# Ensure project root is on sys.path so 'severity.si_utils' is importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from severity.si_utils import (
    CLASS_NAMES as RAW_CLASS_NAMES, 
    CLASS_WEIGHTS, 
    calculate_severity_index,
    grade_severity
)

# API presentation layer: Add " Crack" suffix to match original output
CLASS_NAMES = {
    0: f"{RAW_CLASS_NAMES[0]} Crack", 
    1: f"{RAW_CLASS_NAMES[1]} Crack", 
    2: f"{RAW_CLASS_NAMES[2]} Crack", 
    3: RAW_CLASS_NAMES[3]
}

BBOX_COLORS = {
    0: (52, 152, 219),   # blue
    1: (46, 204, 191),   # teal
    2: (230, 126, 34),   # orange
    3: (231, 76, 60),    # red
}

WEIGHTS_PATH = Path(__file__).parent / "model" / "weights" / "best.pt"

# ── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(title="SHP API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount(
    "/map",
    StaticFiles(directory="map"),
    name="map"
)

# Load model once at startup
model = YOLO(str(WEIGHTS_PATH))


# ── Helpers ──────────────────────────────────────────────────────────────────


def _read_image(file_bytes: bytes) -> np.ndarray:
    """Decode uploaded bytes into a BGR OpenCV image."""
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def _annotate_image(img: np.ndarray, detections: list) -> np.ndarray:
    """Draw bounding boxes and labels on the image."""
    annotated = img.copy()
    for d in detections:
        x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
        cid = d["class_id"]
        color = BBOX_COLORS.get(cid, (255, 255, 255))
        # BGR for OpenCV
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f'{d["class_name"]} {d["confidence"]:.0%}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return annotated


def _image_to_base64(img: np.ndarray) -> str:
    """Encode BGR image to base64 JPEG string."""
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _run_inference(img: np.ndarray):
    """Run YOLO inference and return detections list + results object."""
    h, w = img.shape[:2]
    frame_area = w * h
    results = model(img, verbose=False)

    si, raw_details = calculate_severity_index(results, frame_area)
    grade = grade_severity(si)
    
    # Map back to API expected keys (with " Crack" suffixes where applicable)
    detections = []
    for d in raw_details:
        cid = d["class_id"]
        # Replace short name with full API name
        d["class_name"] = CLASS_NAMES.get(cid, d["class_name"])
        detections.append(d)

    # Per-class counts
    class_counts = {}
    for name in CLASS_NAMES.values():
        class_counts[name] = 0
    for d in detections:
        class_counts[d["class_name"]] += 1

    return detections, si, grade, class_counts, frame_area

def convert_to_degrees(value):
    d = float(value[0])
    m = float(value[1])
    s = float(value[2])

    return d + (m / 60.0) + (s / 3600.0)


def get_gps_from_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes))
        exif = image._getexif()

        if not exif:
            return None

        gps_info = {}

        for tag, value in exif.items():
            decoded = TAGS.get(tag, tag)

            if decoded == "GPSInfo":
                for gps_tag in value:
                    sub_decoded = GPSTAGS.get(gps_tag, gps_tag)
                    gps_info[sub_decoded] = value[gps_tag]

        if not gps_info:
            return None

        lat = convert_to_degrees(gps_info["GPSLatitude"])
        if gps_info["GPSLatitudeRef"] != "N":
            lat = -lat

        lon = convert_to_degrees(gps_info["GPSLongitude"])
        if gps_info["GPSLongitudeRef"] != "E":
            lon = -lon

        return {
            "latitude": lat,
            "longitude": lon
        }

    except Exception as e:
        print("GPS extraction failed:", e)
        return None

# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "ok", "model": str(WEIGHTS_PATH.name)}


@app.post("/api/detect")
async def detect(file: UploadFile = File(...)):
    """Run detection on an uploaded image. Returns annotated image + detections."""
    contents = await file.read()
    img = _read_image(contents)

    detections, si, grade, class_counts, frame_area = _run_inference(img)
    annotated = _annotate_image(img, detections)

    return {
        "image": _image_to_base64(annotated),
        "detections": detections,
        "class_counts": class_counts,
        "total_detections": len(detections),
        "frame_area": frame_area,
        "image_width": img.shape[1],
        "image_height": img.shape[0],
    }


@app.post("/api/severity")
async def severity(file: UploadFile = File(...)):
    contents = await file.read()

    img = _read_image(contents)

    detections, si, grade, class_counts, frame_area = _run_inference(img)

    annotated = _annotate_image(img, detections)

    try:
        gps = get_gps_from_image(contents)

        if gps:
            generate_map(gps, si, grade)

            map_url = "/map/severity_map.html"
        else:
            map_url = None

    except Exception as e:
        print("Map generation error:", e)
        map_url = None

    return {
        "image": _image_to_base64(annotated),
        "detections": detections,
        "class_counts": class_counts,
        "total_detections": len(detections),
        "severity_index": round(si, 5),
        "grade": grade,
        "frame_area": frame_area,
        "image_width": img.shape[1],
        "image_height": img.shape[0],
        "map_url": map_url
    }