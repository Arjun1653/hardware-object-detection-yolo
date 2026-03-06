"""
YOLOv8m Training Script — v2
- Medium model (25M params) for best accuracy
- Same proven settings as yolov8s that achieved 0.732 mAP50
- Conservative augmentation — no mixup, no rotation
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from ultralytics import YOLO

if not torch.cuda.is_available():
    raise SystemExit("\nERROR: CUDA not available!")

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
RUN_NAME    = "hardware_m_v2"

MODEL_SIZE  = "yolov8m.pt"   # Medium — 25M parameters
EPOCHS      = 150
BATCH_SIZE  = 16             # Safe for 6GB VRAM with medium model
IMG_SIZE    = 640
WORKERS     = 2              # Windows: keep at 2
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

        # ── PROVEN SETTINGS (same as yolov8s run) ─────
        cache     = "disk",
        amp       = True,
        optimizer = "SGD",       # SGD — faster convergence than AdamW here
        cos_lr    = False,       # flat LR — simpler, works better for this dataset
        patience  = 20,          # stop if no improvement for 20 epochs

        # ── CONSERVATIVE AUGMENTATION ─────────────────
        mosaic    = 0.5,         # 50% mosaic — same as yolov8s run
        mixup     = 0.0,         # OFF — was hurting performance in v1
        degrees   = 0.0,         # OFF — was hurting performance in v1
        translate = 0.1,
        scale     = 0.5,
        fliplr    = 0.5,
        flipud    = 0.0,
        hsv_h     = 0.015,
        hsv_s     = 0.7,
        hsv_v     = 0.4,

        plots     = True,
        save      = True,
        val       = True,
    )

    best = os.path.join(PROJECT_DIR, RUN_NAME, "weights", "best.pt")
    print(f"\n✓ Training complete!")
    print(f"  Best weights: {best}")


if __name__ == "__main__":
    train()
