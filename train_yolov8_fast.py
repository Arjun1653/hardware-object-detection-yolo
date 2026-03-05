"""
YOLOv8 FAST Training Script — Fixed
- Forces GPU usage
- Fixed workers (Windows crash fix)
- Fixed cache (not enough RAM for 'ram' mode)
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # Force GPU 0 BEFORE torch loads

import torch
from ultralytics import YOLO

# ── Verify GPU before doing anything else ─────────────
if not torch.cuda.is_available():
    raise SystemExit(
        "\nERROR: CUDA not available!\n"
        "Check: nvidia-smi in terminal, and that torch+cu121 is installed.\n"
        "Run:   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
    )

print("=" * 55)
print(f"  GPU  : {torch.cuda.get_device_name(0)}")
print(f"  VRAM : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"  CUDA : {torch.version.cuda}")
print(f"  Torch: {torch.__version__}")
print("=" * 55)

# ─── CONFIG ───────────────────────────────────────────
DATA_ROOT   = r"C:\Users\Arjun\Vision_2\Data"
DATA_YAML   = os.path.join(DATA_ROOT, "data.yaml")
PROJECT_DIR = r"C:\Users\Arjun\Vision_2\runs"
RUN_NAME    = "hardware_s_v1"

MODEL_SIZE  = "yolov8s.pt"   # Small — better accuracy
EPOCHS      = 100
BATCH_SIZE  = 32             # Reduced from 64 — yolov8s is larger, needs more VRAM per image
IMG_SIZE    = 640            # Full resolution — helps detect small nuts and washers
WORKERS     = 2              # MUST be 2 on Windows — higher values cause shared memory crash
# ──────────────────────────────────────────────────────


def clear_cache():
    """Delete stale .cache files from previous crashed runs."""
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
        device    = 0,           # Explicit GPU index
        workers   = WORKERS,
        project   = PROJECT_DIR,
        name      = RUN_NAME,
        exist_ok  = True,

        # ── SPEED ─────────────────────────────────────
        cache     = "disk",      # RAM needs 6.5GB but you only have 4.9GB free
                                 # disk cache still speeds up epoch 2+ significantly
        amp       = True,        # Mixed precision — ~30% faster, half VRAM usage
        patience  = 15,          # Early stopping
        optimizer = "SGD",

        # ── REDUCED AUGMENTATION ──────────────────────
        mosaic    = 0.5,
        mixup     = 0.0,
        degrees   = 0.0,
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
