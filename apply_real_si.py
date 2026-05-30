import json

notebook_path = "severity/severity_index.ipynb"

cells = []

# ── Cell 0: Title & YOLO structure explanation (markdown) ────────────────────
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# Road Damage Severity Index (SI)\n",
        "This notebook calculates a **Road Severity Index** from real YOLOv8 inference outputs, validates it across sample frames, and visualises the results.\n",
        "\n",
        "## 1. YOLOv8 Inference Output Structure\n",
        "When we run `results = model.predict(image)`, YOLOv8 returns a list of `Results` objects one for each input image.\n",
        "\n",
        "The `Results` object contains different attributes depending on the task. For object detection, the primary attribute is `results.boxes`, which is a specialized `Boxes` object containing all detected bounding boxes. \n",
        "\n",
        "Inside `results.boxes`, I can extract the following:\n",
        "- `boxes.xyxy`: This bounds box coordinates in `[x1, y1, x2, y2]` format.\n",
        "- `boxes.conf`: This gives the confidence score.\n",
        "- `boxes.cls`: Basically gives the class ID for the detected object."
    ]
})

# ── Cell 1: Formula explanation (markdown) ───────────────────────────────────
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 2. Severity Index Formula\n",
        "\n",
        "To calculate a Severity Index (SI) that quantifies the overall damage on a road given a single frame, we need to balance the severity of the damage type, the model's confidence, and the physical footprint of the damage relative to the camera view.\n",
        "\n",
        "**Weights ($W$)**\n",
        "- Potholes: `1.0` indicates highest sevearity\n",
        "- Alligator Cracks: `0.8` High sevearity\n",
        "- Longitudinal Cracks: `0.5` Medium severity\n",
        "- Transverse Cracks: `0.3` Low severity\n",
        "\n",
        "**Formula**\n",
        "The formula for severity index that I am going to be using is:\n",
        "$$SI = \\sum_{i=1}^{n} (W_i \\times Confidence_i \\times A_{rel, i})$$\n",
        "\n",
        "Where $A_{rel}$ is the bounding box area relative to the total frame area.\n",
        "\n",
        "**Severity Grading**\n",
        "\n",
        "| Grade | SI Range | Description |\n",
        "|-------|----------|-------------|\n",
        "| Good | SI < 0.005 | Minimal / no visible damage |\n",
        "| Fair | 0.005 ≤ SI < 0.02 | Minor cracks, low urgency |\n",
        "| Poor | 0.02 ≤ SI < 0.05 | Noticeable damage, maintenance needed |\n",
        "| Critical | SI ≥ 0.05 | Severe damage, immediate attention |"
    ]
})

# ── Cell 2: Imports & constants (code) ───────────────────────────────────────
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import os, glob, random\n",
        "import cv2\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import matplotlib.patches as patches\n",
        "from ultralytics import YOLO\n",
        "\n",
        "# ── Constants ──\n",
        "CLASS_NAMES = {0: 'Longitudinal', 1: 'Transverse', 2: 'Alligator', 3: 'Pothole'}\n",
        "CLASS_WEIGHTS = {0: 0.5, 1: 0.3, 2: 0.8, 3: 1.0}\n",
        "CLASS_COLORS = {0: '#3498db', 1: '#2ecc71', 2: '#e67e22', 3: '#e74c3c'}\n",
        "\n",
        "GRADE_THRESHOLDS = [\n",
        "    (0.005, 'Good'),\n",
        "    (0.02,  'Fair'),\n",
        "    (0.05,  'Poor'),\n",
        "    (float('inf'), 'Critical'),\n",
        "]\n",
        "\n",
        "def grade_severity(si):\n",
        "    \"\"\"Map a raw SI score to a human-readable grade.\"\"\"\n",
        "    for threshold, label in GRADE_THRESHOLDS:\n",
        "        if si < threshold:\n",
        "            return label\n",
        "    return 'Critical'\n",
        "\n",
        "print('Constants loaded.')\n",
        "print(f'Classes: {CLASS_NAMES}')\n",
        "print(f'Weights: {CLASS_WEIGHTS}')"
    ]
})

# ── Cell 3: SI calculation function (code) ───────────────────────────────────
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "def calculate_severity_index(results, frame_area, weights=None):\n",
        "    \"\"\"Calculate the Severity Index from YOLOv8 Results objects.\n",
        "    \n",
        "    Args:\n",
        "        results: List of ultralytics Results objects from model().\n",
        "        frame_area: Total pixel area of the frame (width * height).\n",
        "        weights: Optional dict mapping class_id -> severity weight.\n",
        "    \n",
        "    Returns:\n",
        "        Tuple of (severity_index, details_list).\n",
        "        Each detail dict contains class_id, class_name, confidence,\n",
        "        bbox_area, relative_area, and weighted_contribution.\n",
        "    \"\"\"\n",
        "    if weights is None:\n",
        "        weights = CLASS_WEIGHTS\n",
        "    \n",
        "    severity_index = 0.0\n",
        "    details = []\n",
        "    \n",
        "    for r in results:\n",
        "        if r.boxes is None or len(r.boxes) == 0:\n",
        "            continue\n",
        "        \n",
        "        for box in r.boxes:\n",
        "            class_id = int(box.cls)\n",
        "            conf = float(box.conf)\n",
        "            x1, y1, x2, y2 = box.xyxy[0].tolist()\n",
        "            \n",
        "            bbox_area = (x2 - x1) * (y2 - y1)\n",
        "            relative_area = bbox_area / frame_area\n",
        "            \n",
        "            w = weights.get(class_id, 0.1)\n",
        "            contribution = w * conf * relative_area\n",
        "            severity_index += contribution\n",
        "            \n",
        "            details.append({\n",
        "                'class_id': class_id,\n",
        "                'class_name': CLASS_NAMES.get(class_id, f'Unknown({class_id})'),\n",
        "                'confidence': round(conf, 4),\n",
        "                'bbox': [round(v, 1) for v in [x1, y1, x2, y2]],\n",
        "                'bbox_area_px': round(bbox_area, 1),\n",
        "                'relative_area': round(relative_area, 6),\n",
        "                'weight': w,\n",
        "                'contribution': round(contribution, 6),\n",
        "            })\n",
        "    \n",
        "    return severity_index, details\n",
        "\n",
        "print('calculate_severity_index() defined.')"
    ]
})

# ── Cell 4: Load model, single-frame demo (code) ────────────────────────────
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ── Load the trained model ──\n",
        "model_path = '../backend/model/weights/best.pt'\n",
        "if not os.path.exists(model_path):\n",
        "    model_path = 'yolov8n.pt'\n",
        "print(f'Loading model: {model_path}')\n",
        "model = YOLO(model_path)\n",
        "\n",
        "# ── Single-frame demo ──\n",
        "sample_images = sorted(glob.glob('../RDD_SPLIT/val/images/*.jpg'))\n",
        "if not sample_images:\n",
        "    sample_images = sorted(glob.glob('../RDD_SPLIT/train/images/*.jpg'))\n",
        "\n",
        "demo_path = sample_images[0]\n",
        "img = cv2.imread(demo_path)\n",
        "h, w = img.shape[:2]\n",
        "frame_area = w * h\n",
        "\n",
        "results = model(demo_path, verbose=False)\n",
        "si, details = calculate_severity_index(results, frame_area)\n",
        "\n",
        "print(f'\\nImage: {os.path.basename(demo_path)}')\n",
        "print(f'Dimensions: {w}×{h}  |  Frame area: {frame_area:,} px²')\n",
        "print(f'Detections: {len(details)}')\n",
        "print(f'\\n{\"#\":<3} {\"Class\":<15} {\"Conf\":>6} {\"BBox Area\":>10} {\"Rel Area\":>10} {\"Weight\":>6} {\"Contrib\":>10}')\n",
        "print('─' * 68)\n",
        "for i, d in enumerate(details, 1):\n",
        "    print(f'{i:<3} {d[\"class_name\"]:<15} {d[\"confidence\"]:>6.4f} {d[\"bbox_area_px\"]:>10.1f} {d[\"relative_area\"]:>10.6f} {d[\"weight\"]:>6.1f} {d[\"contribution\"]:>10.6f}')\n",
        "\n",
        "print(f'\\n{\"═\" * 68}')\n",
        "print(f'Severity Index: {si:.5f}  →  Grade: {grade_severity(si)}')"
    ]
})

# ── Cell 5: Markdown – Batch Validation header ──────────────────────────────
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 3. Batch Validation\n",
        "Run inference on a random sample of **30 frames** from the validation set (or training set if val is small) and compute the SI for each. This produces a results table and summary statistics."
    ]
})

# ── Cell 6: Batch inference (code) ──────────────────────────────────────────
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ── Batch validation ──\n",
        "all_images = sorted(glob.glob('../RDD_SPLIT/val/images/*.jpg'))\n",
        "if len(all_images) < 10:\n",
        "    all_images = sorted(glob.glob('../RDD_SPLIT/train/images/*.jpg'))\n",
        "\n",
        "random.seed(42)\n",
        "sample_paths = random.sample(all_images, min(30, len(all_images)))\n",
        "\n",
        "rows = []\n",
        "for img_path in sample_paths:\n",
        "    img = cv2.imread(img_path)\n",
        "    h, w = img.shape[:2]\n",
        "    fa = w * h\n",
        "    \n",
        "    res = model(img_path, verbose=False)\n",
        "    si, det = calculate_severity_index(res, fa)\n",
        "    \n",
        "    # Per-class detection counts\n",
        "    class_counts = {CLASS_NAMES[k]: 0 for k in CLASS_NAMES}\n",
        "    for d in det:\n",
        "        class_counts[d['class_name']] = class_counts.get(d['class_name'], 0) + 1\n",
        "    \n",
        "    row = {\n",
        "        'image': os.path.basename(img_path),\n",
        "        'width': w,\n",
        "        'height': h,\n",
        "        'detections': len(det),\n",
        "        **class_counts,\n",
        "        'SI': round(si, 5),\n",
        "        'grade': grade_severity(si),\n",
        "    }\n",
        "    rows.append(row)\n",
        "\n",
        "df = pd.DataFrame(rows)\n",
        "print(f'Processed {len(df)} frames.\\n')\n",
        "df.sort_values('SI', ascending=False, inplace=True)\n",
        "df.reset_index(drop=True, inplace=True)\n",
        "df.index += 1\n",
        "df.index.name = '#'\n",
        "print(df.to_string())"
    ]
})

# ── Cell 7: Summary statistics (code) ───────────────────────────────────────
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ── Summary statistics ──\n",
        "print('Severity Index Statistics')\n",
        "print('═' * 35)\n",
        "print(f'  Mean SI:   {df[\"SI\"].mean():.5f}')\n",
        "print(f'  Median SI: {df[\"SI\"].median():.5f}')\n",
        "print(f'  Std Dev:   {df[\"SI\"].std():.5f}')\n",
        "print(f'  Min SI:    {df[\"SI\"].min():.5f}  ({df.loc[df[\"SI\"].idxmin(), \"image\"]})')\n",
        "print(f'  Max SI:    {df[\"SI\"].max():.5f}  ({df.loc[df[\"SI\"].idxmax(), \"image\"]})')\n",
        "\n",
        "print(f'\\nGrade Distribution')\n",
        "print('═' * 35)\n",
        "grade_counts = df['grade'].value_counts()\n",
        "for grade in ['Good', 'Fair', 'Poor', 'Critical']:\n",
        "    count = grade_counts.get(grade, 0)\n",
        "    pct = count / len(df) * 100\n",
        "    bar = '█' * int(pct / 2)\n",
        "    print(f'  {grade:<10} {count:>3}  ({pct:5.1f}%)  {bar}')\n",
        "\n",
        "print(f'\\nTotal detections across all frames: {df[\"detections\"].sum()}')"
    ]
})

# ── Cell 8: Markdown – Visualisations header ─────────────────────────────────
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 4. Visualisations"
    ]
})

# ── Cell 9: SI Distribution Histogram (code) ────────────────────────────────
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ── SI Distribution ──\n",
        "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
        "\n",
        "# Histogram\n",
        "ax = axes[0]\n",
        "grade_colors = {'Good': '#2ecc71', 'Fair': '#f1c40f', 'Poor': '#e67e22', 'Critical': '#e74c3c'}\n",
        "colors = [grade_colors[g] for g in df['grade']]\n",
        "\n",
        "ax.bar(range(len(df)), df['SI'].values, color=colors, edgecolor='white', linewidth=0.5)\n",
        "ax.set_xlabel('Frame (sorted by SI)')\n",
        "ax.set_ylabel('Severity Index')\n",
        "ax.set_title('Severity Index per Frame')\n",
        "\n",
        "# Add grade threshold lines\n",
        "for thresh, label in GRADE_THRESHOLDS[:-1]:\n",
        "    ax.axhline(y=thresh, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)\n",
        "    ax.text(len(df) + 0.3, thresh, label, va='center', fontsize=8, color='gray')\n",
        "\n",
        "ax.set_xticks([])\n",
        "\n",
        "# Grade pie chart\n",
        "ax2 = axes[1]\n",
        "grade_order = ['Good', 'Fair', 'Poor', 'Critical']\n",
        "counts = [grade_counts.get(g, 0) for g in grade_order]\n",
        "pie_colors = [grade_colors[g] for g in grade_order]\n",
        "nonzero = [(g, c, col) for g, c, col in zip(grade_order, counts, pie_colors) if c > 0]\n",
        "\n",
        "if nonzero:\n",
        "    labels, vals, cols = zip(*nonzero)\n",
        "    ax2.pie(vals, labels=labels, colors=cols, autopct='%1.0f%%',\n",
        "            startangle=90, textprops={'fontsize': 10})\n",
        "    ax2.set_title('Grade Distribution')\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.savefig('si_distribution.png', dpi=150, bbox_inches='tight')\n",
        "plt.show()\n",
        "print('Saved: si_distribution.png')"
    ]
})

# ── Cell 10: Per-class breakdown (code) ──────────────────────────────────────
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ── Per-class damage breakdown ──\n",
        "class_cols = list(CLASS_NAMES.values())\n",
        "class_totals = df[class_cols].sum()\n",
        "\n",
        "fig, ax = plt.subplots(figsize=(8, 5))\n",
        "bar_colors = [CLASS_COLORS[i] for i in range(len(class_cols))]\n",
        "bars = ax.barh(class_cols, class_totals.values, color=bar_colors, edgecolor='white')\n",
        "\n",
        "for bar, val in zip(bars, class_totals.values):\n",
        "    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,\n",
        "            f'{int(val)}', va='center', fontweight='bold')\n",
        "\n",
        "ax.set_xlabel('Total Detections (across all sampled frames)')\n",
        "ax.set_title('Damage Type Distribution')\n",
        "ax.invert_yaxis()\n",
        "plt.tight_layout()\n",
        "plt.savefig('damage_breakdown.png', dpi=150, bbox_inches='tight')\n",
        "plt.show()\n",
        "print('Saved: damage_breakdown.png')"
    ]
})

# ── Cell 11: Annotated detection visualisation (code) ────────────────────────
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ── Annotated detections on worst frames ──\n",
        "worst_frames = df.head(4)\n",
        "\n",
        "fig, axes = plt.subplots(2, 2, figsize=(14, 10))\n",
        "axes = axes.flatten()\n",
        "\n",
        "for idx, (_, row) in enumerate(worst_frames.iterrows()):\n",
        "    ax = axes[idx]\n",
        "    img_path = None\n",
        "    for base in ['../RDD_SPLIT/val/images/', '../RDD_SPLIT/train/images/']:\n",
        "        candidate = os.path.join(base, row['image'])\n",
        "        if os.path.exists(candidate):\n",
        "            img_path = candidate\n",
        "            break\n",
        "    \n",
        "    if img_path is None:\n",
        "        ax.set_visible(False)\n",
        "        continue\n",
        "    \n",
        "    img = cv2.imread(img_path)\n",
        "    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\n",
        "    h, w = img.shape[:2]\n",
        "    fa = w * h\n",
        "    \n",
        "    ax.imshow(img_rgb)\n",
        "    \n",
        "    res = model(img_path, verbose=False)\n",
        "    for r in res:\n",
        "        if r.boxes is None:\n",
        "            continue\n",
        "        for box in r.boxes:\n",
        "            cid = int(box.cls)\n",
        "            conf = float(box.conf)\n",
        "            x1, y1, x2, y2 = box.xyxy[0].tolist()\n",
        "            color = CLASS_COLORS.get(cid, '#ffffff')\n",
        "            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,\n",
        "                                     linewidth=2, edgecolor=color, facecolor='none')\n",
        "            ax.add_patch(rect)\n",
        "            ax.text(x1, y1 - 4, f'{CLASS_NAMES[cid]} {conf:.2f}',\n",
        "                    fontsize=7, color='white', fontweight='bold',\n",
        "                    bbox=dict(facecolor=color, alpha=0.7, pad=1, edgecolor='none'))\n",
        "    \n",
        "    ax.set_title(f'{row[\"image\"]}\\nSI={row[\"SI\"]:.5f} ({row[\"grade\"]})', fontsize=9)\n",
        "    ax.axis('off')\n",
        "\n",
        "plt.suptitle('Top 4 Worst Frames by Severity Index', fontsize=13, fontweight='bold')\n",
        "plt.tight_layout()\n",
        "plt.savefig('worst_frames.png', dpi=150, bbox_inches='tight')\n",
        "plt.show()\n",
        "print('Saved: worst_frames.png')"
    ]
})

# ── Assemble notebook ────────────────────────────────────────────────────────
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "venv",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.13.13"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open(notebook_path, "w") as f:
    json.dump(nb, f, indent=1)

print(f"Wrote {len(cells)} cells to {notebook_path}")
