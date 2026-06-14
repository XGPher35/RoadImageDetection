# Deep Learning-Based Road Damage Detection and Severity Prioritization

An automated, camera-based pipeline that detects road defects with **YOLOv8**, scores
each frame with a composite **Road Severity Index (SI)**, and visualises the results on
an interactive **Folium** map for maintenance prioritisation.

Trained on the **Road Damage Dataset 2022 (RDD2022)** across four defect classes:
longitudinal cracks, transverse cracks, alligator cracks, and potholes.

**Final model (YOLOv8s, 50 epochs):** mAP@50 **57.5%** · mAP@50–95 **30.1%** ·
Precision **61.3%** · Recall **55.1%**.

---

## Project Structure

```
RoadImageDetection/
├── pipeline.py              # ★ Master integration: images → SI → ranked CSV → map
├── evaluate.py              # ★ Option A re-evaluation (hi-res + TTA, no retraining)
├── generate_csv.py          # Human-validation template generator
├── apply_real_si.py         # Builds the severity_index notebook
│
├── backend/                 # Data + model + API + map backend
│   ├── audit_and_split.py       # (Rubin) dataset audit + stratified 80/10/10 split
│   ├── augmentation_pipeline.py # (Rubin) Albumentations augmentation pipeline
│   ├── severity_map.py          # (Sandesh) Folium map generation
│   ├── severity_points.json     # (Sandesh) geocoded detection store
│   ├── api.py                   # FastAPI service
│   ├── requirements.txt
│   └── model/                   # (Arjit) training + weights
│       ├── train_colab.ipynb    # YOLOv8 training notebook (Colab)
│       ├── rdd2022.yaml         # dataset config
│       ├── predict_sample.py    # minimal inference example
│       └── weights/best.pt      # final trained model
│
├── severity/                # (Adarsha) Severity Index
│   ├── si_utils.py              # single source of truth: weights, grading, smoothing
│   ├── generate_segment_report.py  # per-folder ranked CSV
│   ├── severity_index.ipynb     # SI derivation + validation notebook
│   └── *.png                    # validation / sensitivity / distribution figures
│
├── RDD2022_runs/            # training runs + validation outputs (curves, matrices)
├── frontend/                # Vite + React frontend (separate app)
└── main.tex                 # final project report (LaTeX)
```

## Team / Domain Ownership

| Member  | Domain   | Key files |
|---------|----------|-----------|
| Rubin   | Data     | `backend/audit_and_split.py`, `backend/augmentation_pipeline.py` |
| Arjit   | Model    | `backend/model/`, `train_colab.ipynb`, `weights/best.pt` |
| Adarsha | Severity | `severity/si_utils.py`, `severity/severity_index.ipynb` |
| Sandesh | Maps     | `backend/severity_map.py`, `backend/map/` |

---

## Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r backend/requirements.txt
```

> **Note:** `backend/requirements.txt` pins `torch==2.3.0`, which predates Python 3.14.
> On Python 3.14, install an unpinned current `torch` (≥2.12) instead — verified working
> with `ultralytics` + `torch 2.12` on 3.14.

---

## Usage

### 1. Train the model (Colab / GPU)

Open `backend/model/train_colab.ipynb`, or run directly:

```bash
yolo train model=yolov8s.pt data=backend/model/rdd2022.yaml \
    epochs=50 imgsz=640 batch=16 lr0=0.01 cos_lr=True
```

### 2. Evaluate — with Option A inference-time boost (no retraining)

Re-validates the trained weights at higher resolution with test-time augmentation,
targeting recall on the thin-crack classes. Regenerates all PR/F1/P/R curves and the
confusion matrix.

```bash
python evaluate.py --model backend/model/weights/best.pt \
    --data backend/model/rdd2022.yaml --imgsz 1280
```

Outputs metrics to stdout and figures to `runs/detect/val_optionA/`.

### 3. Run the full pipeline

End-to-end: YOLO inference → per-frame SI → EWMA smoothing → attach GPS coordinates →
ranked CSV → interactive map.

```bash
python pipeline.py --images_dir <segment_frames> \
    --model backend/model/weights/best.pt \
    --coords backend/severity_points.json \
    --csv_out ranked_severity.csv \
    --map_out maps/severity_map.html
```

Produces `ranked_severity.csv` (segments ranked worst-first) and an interactive
`severity_map.html`. Open the HTML in any browser to view the colour-coded markers,
heatmap, and layer toggles.

### 4. Frontend

```bash
cd frontend
npm install
npm run dev
```

---

## Severity Index

Each frame is scored as:

```
SI = Σ ( W_i × Confidence_i × A_rel,i )
```

where `W` is the class severity weight (Pothole 1.0, Alligator 0.8, Longitudinal 0.5,
Transverse 0.3), `Confidence` is the detection confidence, and `A_rel` is the
bounding-box area relative to the frame. Per-frame scores are smoothed (EWMA, α=0.3)
into segment scores and graded:

| Grade | SI Range | Map Colour |
|-------|----------|------------|
| Good     | SI < 0.005          | Green       |
| Fair     | 0.005 ≤ SI < 0.02   | Light Green |
| Poor     | 0.02 ≤ SI < 0.05    | Orange      |
| Critical | SI ≥ 0.05           | Red         |

---

## Scope Note

The project is trained and evaluated entirely on **RDD2022**. Collection/annotation of a
local Nepali road dataset and fine-tuning to close the geographic domain gap are
identified as **future work**, not part of this iteration.
```