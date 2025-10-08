# --- MAX PERFORMANCE TRAINING PARAMETERS ---
# This script is designed for a final, high-performance training run.
# It uses a larger model and longer training time to achieve the best possible score.

MODEL_SIZE = 'yolov8m.pt' # Using the MEDIUM model for higher accuracy.
EPOCHS = 100         # Increased significantly for the larger model.
IMAGE_SIZE = 736     # Train on slightly larger images for more detail.

# --- STABILITY & RESOURCE MANAGEMENT for a LARGER MODEL ---
BATCH_SIZE = 2       # REDUCED to 2. This is critical to fit the larger model on a 4GB GPU.
WORKERS = 2          
CACHE_DATA = False   
SAVE_PERIOD = 10     # Save a checkpoint every 10 epochs.

# --- HYPERPARAMETERS & AUGMENTATIONS ---
OPTIMIZER = 'AdamW'
LR0 = 0.0008         # Slightly reduced learning rate for stable training.
MOSAIC = 0.8         
MIXUP = 0.15      
COPY_PASTE = 0.15 
FLIPUD = 0.5     
FLIPLR = 0.5     

# --- NAMING ---
PROJECT_NAME = 'runs/detect' 
# A unique name will be generated for each new run using a timestamp.

import argparse
from ultralytics import YOLO
import os
import sys
import datetime
import glob
import pandas as pd

def log(message):
    """Logs a message to the console and a file with a timestamp."""
    now = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    print(f"{now} {message}")
    with open("train_log.txt", "a") as f:
        f.write(f"{now} {message}\n")

def find_latest_run_for_resume():
    """Finds the most recent 'last.pt' checkpoint file to resume training from."""
    run_dirs = glob.glob(os.path.join(PROJECT_NAME, '*'))
    run_dirs = [d for d in run_dirs if os.path.isdir(d)]
    
    if not run_dirs:
        return None, False
        
    latest_dir = max(run_dirs, key=os.path.getmtime)
    
    # Check if the training is already complete
    results_file = os.path.join(latest_dir, 'results.csv')
    if os.path.exists(results_file):
        try:
            df = pd.read_csv(results_file)
            # The column name for epoch is 'epoch' (1-indexed)
            # Check if the number of epochs in the log is one less than the target
            if not df.empty and (df['epoch'].max() >= EPOCHS):
                log(f"Latest run '{os.path.basename(latest_dir)}' has already completed {df['epoch'].max()}/{EPOCHS} epochs.")
                return None, True # Indicate that the last run was finished
        except Exception as e:
            log(f"Could not read results.csv from {latest_dir}: {e}")

    weights_path = os.path.join(latest_dir, 'weights', 'last.pt')
    if os.path.exists(weights_path):
        log(f"Found checkpoint to resume from: {weights_path}")
        return weights_path, False # Indicate that the last run was not finished
    
    return None, False

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description="Professional YOLOv8 Training Script")
    parser.add_argument('--resume', action='store_true', help='Resume from the last available checkpoint.')
    args = parser.parse_args()

    this_dir = os.path.dirname(__file__)
    os.chdir(this_dir)
    
    model_path = MODEL_SIZE
    run_name = f"final_run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if args.resume:
        model_path, is_finished = find_latest_run_for_resume()
        if is_finished:
            log("Previous training was already complete. Starting a new run.")
            model_path = MODEL_SIZE
        elif model_path:
            log(f"Resuming training session from {model_path}")
            # Use the same name as the run we are resuming from
            run_name = os.path.basename(os.path.dirname(os.path.dirname(model_path)))
        else:
            log("No checkpoint found. Starting a new training session.")
            model_path = MODEL_SIZE
    else:
        log(f"Starting a new training session with model: {MODEL_SIZE}")

    log(f"Run name: {run_name}")
    log(f"Training for {EPOCHS} epochs with a batch size of {BATCH_SIZE} and image size {IMAGE_SIZE}.")
    
    model = YOLO(model_path)
    
    # --- START TRAINING ---
    results = model.train(
        data=os.path.join(this_dir, "yolo_params.yaml"), 
        epochs=EPOCHS,
        device=0,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        workers=WORKERS,
        cache=CACHE_DATA,
        project=PROJECT_NAME,
        name=run_name,
        save_period=SAVE_PERIOD,
        exist_ok=True,
        optimizer=OPTIMIZER,
        lr0=LR0,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=10.0, translate=0.1, scale=0.5, shear=5.0,
        perspective=0.0, flipud=FLIPUD, fliplr=FLIPLR,
        mosaic=MOSAIC, mixup=MIXUP, copy_paste=COPY_PASTE
    )
    
    log("Training session completed!")
    log(f"Your best model and results are saved in: {os.path.join(PROJECT_NAME, run_name)}")
