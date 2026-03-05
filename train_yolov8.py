"""
YOLOv8 Object Detection Training Script
- Windows compatible
- CUDA GPU accelerated
- Polygon (segmentation) label format supported
"""

import os
import sys
import torch
from pathlib import Path
from ultralytics import YOLO

# ─────────────────────────────────────────────
# CONFIG — edit these paths to match your setup
# ─────────────────────────────────────────────
DATA_ROOT = r"C:\Users\Arjun\Vision_2\Data"   # Root folder containing train/valid/test
DATA_YAML  = os.path.join(DATA_ROOT, "data.yaml")  # Path to data.yaml

MODEL_SIZE = "yolov8m.pt"   # Options: yolov8n / yolov8s / yolov8m / yolov8l / yolov8x
                             # 'n' = fastest/lightest, 'x' = most accurate/heaviest

EPOCHS      = 100
BATCH_SIZE  = 16       # Reduce to 8 if you get CUDA out-of-memory errors
IMG_SIZE    = 640      # Standard YOLO input resolution
WORKERS     = 4        # Dataloader workers (keep ≤4 on Windows to avoid issues)
PROJECT_DIR = r"C:\Users\Arjun\Vision_2\runs"  # Where results are saved
RUN_NAME    = "hardware_detection_v1"

# ─────────────────────────────────────────────
# GPU CHECK
# ─────────────────────────────────────────────
def check_gpu():
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. Training will use CPU (much slower).")
        return "cpu"
    
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✓ GPU detected : {gpu_name}")
    print(f"✓ VRAM         : {vram_gb:.1f} GB")
    
    # Auto-adjust batch size for low-VRAM GPUs
    global BATCH_SIZE
    if vram_gb < 4 and BATCH_SIZE > 8:
        BATCH_SIZE = 4
        print(f"  Low VRAM detected — batch size reduced to {BATCH_SIZE}")
    elif vram_gb < 8 and BATCH_SIZE > 16:
        BATCH_SIZE = 8
        print(f"  Adjusting batch size to {BATCH_SIZE} for available VRAM")
    
    return 0  # device index for CUDA


# ─────────────────────────────────────────────
# VALIDATE data.yaml EXISTS
# ─────────────────────────────────────────────
def validate_setup():
    if not os.path.exists(DATA_YAML):
        print(f"\nERROR: data.yaml not found at:\n  {DATA_YAML}")
        print("\nMake sure your data.yaml looks like this:\n")
        print(EXAMPLE_YAML)
        sys.exit(1)
    
    # Check train/valid folders
    for split in ["train", "valid", "test"]:
        img_dir = os.path.join(DATA_ROOT, split, "images")
        lbl_dir = os.path.join(DATA_ROOT, split, "labels")
        if os.path.exists(img_dir):
            n_imgs = len([f for f in os.listdir(img_dir)
                          if f.lower().endswith((".jpg", ".jpeg", ".png"))])
            print(f"✓ {split:6s} images: {n_imgs}")
        else:
            print(f"  {split:6s} images folder not found: {img_dir}")


EXAMPLE_YAML = """
path: C:/Users/Arjun/Vision_2/Data   # ← your DATA_ROOT (use forward slashes)
train: train/images
val:   valid/images
test:  test/images

nc: 4   # ← number of classes (adjust to match your dataset)
names:
  0: nail
  1: bolt
  2: screw
  3: washer
  # Replace with your actual class names from README.dataset or data.yaml
"""


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
def train():
    device = check_gpu()
    validate_setup()

    print(f"\n{'─'*50}")
    print(f"  Model      : {MODEL_SIZE}")
    print(f"  Epochs     : {EPOCHS}")
    print(f"  Batch size : {BATCH_SIZE}")
    print(f"  Image size : {IMG_SIZE}")
    print(f"  Device     : {'GPU (CUDA:0)' if device == 0 else 'CPU'}")
    print(f"  Output     : {PROJECT_DIR}/{RUN_NAME}")
    print(f"{'─'*50}\n")

    # Load pretrained YOLOv8 model
    model = YOLO(MODEL_SIZE)

    # Train
    results = model.train(
        data       = DATA_YAML,
        epochs     = EPOCHS,
        batch      = BATCH_SIZE,
        imgsz      = IMG_SIZE,
        device     = device,
        workers    = WORKERS,
        project    = PROJECT_DIR,
        name       = RUN_NAME,
        exist_ok   = True,       # Overwrite existing run folder
        patience   = 20,         # Early stopping: stop if no improvement for 20 epochs
        save       = True,
        plots      = True,       # Save training plots
        val        = True,       # Validate after each epoch
        cache      = False,      # Set True to cache images in RAM (speeds up if you have lots of RAM)
        amp        = True,       # Mixed precision (faster + less VRAM)
        optimizer  = "AdamW",
        lr0        = 0.001,
        weight_decay = 0.0005,
        mosaic     = 1.0,        # Mosaic augmentation
        degrees    = 10.0,       # Rotation augmentation
        flipud     = 0.1,
        fliplr     = 0.5,
    )

    print("\n✓ Training complete!")
    print(f"  Best weights : {PROJECT_DIR}\\{RUN_NAME}\\weights\\best.pt")
    print(f"  Last weights : {PROJECT_DIR}\\{RUN_NAME}\\weights\\last.pt")
    return results


# ─────────────────────────────────────────────
# VALIDATION on test set
# ─────────────────────────────────────────────
def validate_test(weights_path=None):
    if weights_path is None:
        weights_path = os.path.join(PROJECT_DIR, RUN_NAME, "weights", "best.pt")
    
    if not os.path.exists(weights_path):
        print(f"Weights not found at {weights_path}")
        return

    print(f"\nRunning validation on test set with: {weights_path}")
    model = YOLO(weights_path)
    
    metrics = model.val(
        data   = DATA_YAML,
        split  = "test",
        imgsz  = IMG_SIZE,
        device = 0 if torch.cuda.is_available() else "cpu",
        plots  = True,
    )
    
    print(f"\n  mAP50    : {metrics.box.map50:.4f}")
    print(f"  mAP50-95 : {metrics.box.map:.4f}")
    return metrics


# ─────────────────────────────────────────────
# INFERENCE — run on a single image or folder
# ─────────────────────────────────────────────
def predict(source, weights_path=None, conf=0.25):
    """
    source: path to image file, folder, or 0 for webcam
    conf  : confidence threshold (0–1)
    """
    if weights_path is None:
        weights_path = os.path.join(PROJECT_DIR, RUN_NAME, "weights", "best.pt")

    model = YOLO(weights_path)
    results = model.predict(
        source  = source,
        conf    = conf,
        imgsz   = IMG_SIZE,
        device  = 0 if torch.cuda.is_available() else "cpu",
        save    = True,
        project = PROJECT_DIR,
        name    = "predict",
    )
    return results


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # ── Step 1: Train ──
    train()

    # ── Step 2: Evaluate on test set (uncomment after training) ──
    # validate_test()

    # ── Step 3: Run inference on a single image (uncomment to use) ──
    # predict(source=r"C:\path\to\your\image.jpg", conf=0.3)
