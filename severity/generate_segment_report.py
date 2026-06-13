"""
severity/generate_segment_report.py

End-to-End Integration Script for the Road Damage Severity Index pipeline.
This script takes a folder of images representing a "road segment" (e.g. video frames),
runs the YOLOv8 model + SI calculation on each frame, applies temporal smoothing (EWMA),
and outputs a single ranked CSV report.

This CSV report serves as the handoff artifact for mapping integrations (e.g. Folium).

Output CSV Columns:
- image                 : Filename of the frame.
- per_frame_SI          : Raw Severity Index calculated for this single frame.
- grade                 : Grading of the raw SI (Good, Fair, Poor, Critical).
- segment_aggregated_SI : The smoothed SI for the segment (using EWMA). This represents 
                          the aggregated severity of the entire segment up to this frame.
- segment_grade         : Grading of the aggregated SI.
"""

import argparse
import os
import glob
import cv2
import pandas as pd
from ultralytics import YOLO

import sys
# Ensure we can import si_utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from severity.si_utils import calculate_severity_index, grade_severity, temporal_smooth_ewma, temporal_smooth_sma

def main():
    parser = argparse.ArgumentParser(description="Generate a Road Segment Severity Report")
    parser.add_argument('--images_dir', type=str, required=True, help="Directory containing sequence of images")
    parser.add_argument('--model', type=str, required=True, help="Path to YOLOv8 model weights (.pt)")
    parser.add_argument('--method', type=str, choices=['ewma', 'sma'], default='ewma', help="Smoothing method (default: ewma)")
    parser.add_argument('--alpha', type=float, default=0.3, help="EWMA smoothing alpha (default: 0.3)")
    parser.add_argument('--window', type=int, default=5, help="SMA window size (default: 5)")
    parser.add_argument('--output', type=str, default='severity/segment_report.csv', help="Output CSV path")
    args = parser.parse_args()

    if not os.path.isdir(args.images_dir):
        print(f"Error: {args.images_dir} is not a valid directory.")
        return

    image_paths = sorted(glob.glob(os.path.join(args.images_dir, '*.jpg'))) + \
                  sorted(glob.glob(os.path.join(args.images_dir, '*.png')))
    
    if not image_paths:
        print(f"No .jpg or .png images found in {args.images_dir}")
        return

    print(f"Loading model from {args.model}...")
    try:
        model = YOLO(args.model)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print(f"Processing {len(image_paths)} images...")
    
    results_list = []
    raw_si_scores = []
    
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        h, w = img.shape[:2]
        frame_area = w * h
        
        # Run inference
        res = model(img_path, verbose=False)
        si, _ = calculate_severity_index(res, frame_area)
        
        raw_si_scores.append(si)
        results_list.append({
            'image': os.path.basename(img_path),
            'per_frame_SI': round(si, 5),
            'grade': grade_severity(si)
        })

    print(f"Applying temporal smoothing ({args.method})...")
    if args.method == 'ewma':
        smoothed_scores = temporal_smooth_ewma(raw_si_scores, alpha=args.alpha)
    else:
        smoothed_scores = temporal_smooth_sma(raw_si_scores, window_size=args.window)

    # Add smoothed results
    for i, row in enumerate(results_list):
        agg_si = smoothed_scores[i]
        row['segment_aggregated_SI'] = round(agg_si, 5)
        row['segment_grade'] = grade_severity(agg_si)

    df = pd.DataFrame(results_list)
    
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df.to_csv(args.output, index=False)
    
    print(f"\nReport generated successfully: {args.output}")
    print(f"Head:\n{df.head()}")

if __name__ == "__main__":
    main()
