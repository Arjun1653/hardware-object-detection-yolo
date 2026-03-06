"""
YOLOv8 High Accuracy Training Script
- YOLOv8m (medium model)
- Full resolution 640px
- Full augmentation
- Cosine LR + longer training
- All accuracy optimizations applied
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from ultralytics import YOLO

# ── Verify GPU ─────────────────────────────────────────
if not torch.cuda.is_available():
    raise SystemExit(
        "\nERROR: CUDA not available!\n"
        "Run: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
    )

print("=" * 55)
print(f"  GPU  : {torch.cuda.get_device_name(0)}")
print(f"  VRAM : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"  CUDA : {torch.version.cuda}")
print(f"  Torch: {torch.__version__}")
print("=" * 55)

# ─── CONFIG ────────────────────────────────────────────
DATA_ROOT   = r"C:\Users\Arjun\Vision_2\Data"
DATA_YAML   = os.path.join(DATA_ROOT, "data.yaml")
PROJECT_DIR = r"C:\Users\Arjun\Vision_2\runs"
RUN_NAME    = "hardware_m_v1"

MODEL_SIZE  = "yolov8m.pt"   # Medium — best accuracy/speed balance
EPOCHS      = 200            # Long run — early stopping will cut this short
BATCH_SIZE  = 16             # yolov8m is larger, 16 is safe for 6GB VRAM
IMG_SIZE    = 640
WORKERS     = 2              # MUST stay at 2 on Windows
# ───────────────────────────────────────────────────────


def clear_cache():
    for split in ["train", "valid", "test"]:
        cache = os.path.join(DATA_ROOT, split, "labels.cache")
        if os.path.exists(cache):
            os.remove(cache)
            print(f"  Cleared: {cache}")


def train():
    print("\nClearing old cache files...")
    clear_cache()

    model = YOLO(MODEL_SIZE)

    model.train(
        data      = DATA_YAML,
        epochs    = EPOCHS,
        batch     = BATCH_SIZE,
        imgsz     = IMG_SIZE,
        device    = 0,
        workers   = WORKERS,
        project   = PROJECT_DIR,
        name      = RUN_NAME,
        exist_ok  = True,

        # ── ACCURACY OPTIMIZATIONS ─────────────────────
        cache     = "disk",
        amp       = True,
        cos_lr    = True,        # Cosine LR schedule — better convergence than flat LR
        patience  = 30,          # More patience before early stopping
        optimizer = "AdamW",     # AdamW works better than SGD for longer runs

        # ── FULL AUGMENTATION ──────────────────────────
        mosaic    = 1.0,         # Always use mosaic (was 0.5)
        mixup     = 0.1,         # Blend two images together — improves generalization
        degrees   = 15.0,        # Rotate up to 15° — helps washers especially
        translate = 0.2,
        scale     = 0.7,         # More aggressive scaling (was 0.5)
        fliplr    = 0.5,
        flipud    = 0.1,         # Occasional vertical flip (was 0.0)
        hsv_h     = 0.015,
        hsv_s     = 0.7,
        hsv_v     = 0.4,
        erasing   = 0.4,         # Random erasing — forces model to detect partial objects
        crop_fraction = 1.0,

        # ── LOSS WEIGHTS ───────────────────────────────
        box       = 7.5,         # Bounding box loss weight
        cls       = 0.5,         # Classification loss weight
        dfl       = 1.5,         # Distribution focal loss weight

        plots     = True,
        save      = True,
        val       = True,
    )

    best = os.path.join(PROJECT_DIR, RUN_NAME, "weights", "best.pt")
    print(f"\n✓ Training complete!")
    print(f"  Best weights : {best}")


if __name__ == "__main__":
    train()
