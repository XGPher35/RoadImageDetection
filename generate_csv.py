import os
import glob
import cv2
import pandas as pd
import random
import sys

# Ensure we can import si_utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from severity.si_utils import calculate_severity_index, grade_severity
from ultralytics import YOLO

def generate_template():
    csv_path = 'severity/human_validation_template.csv'
    model_path = 'backend/model/weights/best.pt'
    
    print("Generating 50-image dataset for human validation...")
    val_imgs = sorted(glob.glob('RDD_SPLIT/val/images/*.jpg'))
    if not val_imgs:
        val_imgs = sorted(glob.glob('../RDD_SPLIT/val/images/*.jpg'))
    print(f"Found {len(val_imgs)} images.")
    random.seed(123)
    val_sample = random.sample(val_imgs, min(100, len(val_imgs)))
    
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    val_data = []
    for img_path in val_sample:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read {img_path}")
            continue
        res = model(img_path, verbose=False)
        si, _ = calculate_severity_index(res, img.shape[1]*img.shape[0])
        val_data.append({'image': os.path.basename(img_path), 'model_SI': si, 'model_grade': grade_severity(si)})
        
    print(f"Successfully processed {len(val_data)} images.")
    if len(val_data) == 0:
        return
        
    val_df = pd.DataFrame(val_data)
    print("Columns before sampling:", val_df.columns)
    
    # Stratified sampling, with fallback
    try:
        val_df = val_df.groupby('model_grade', group_keys=False).apply(lambda x: x.sample(min(len(x), 15)))
    except Exception as e:
        print(f"Sampling failed: {e}")
        pass
        
    print("Columns after sampling:", val_df.columns)
    if 'model_grade' not in val_df.columns:
        print("model_grade was lost! Restoring from val_data")
        val_df = pd.DataFrame(val_data)
        
    val_df = val_df.sample(min(len(val_df), 50), random_state=42) # Ensure exactly 50 max
    
    # Rank them based on model SI (1 = most severe, highest SI)
    val_df['model_rank'] = val_df['model_SI'].rank(ascending=False, method='min').astype(int)
    val_df.sort_values('model_rank', inplace=True)
    
    val_df['human_score'] = ''
    val_df['human_rank'] = ''
    
    val_df[['image', 'model_SI', 'model_grade', 'model_rank', 'human_score', 'human_rank']].to_csv(csv_path, index=False)
    print(f"Generated template with {len(val_df)} images: {csv_path}")

if __name__ == "__main__":
    generate_template()
