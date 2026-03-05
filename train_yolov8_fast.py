"""
YOLOv8 FAST Training Script
Optimized for speed on RTX 4050 (6GB VRAM)
"""

import os
import torch
from ultralytics import YOLO

# ─── CONFIG ───────────────────────────────────────────
DATA_ROOT   = r"C:\Users\Arjun\Vision_2\Data"
DATA_YAML   = os.path.join(DATA_ROOT, "data.yaml")
PROJECT_DIR = r"C:\Users\Arjun\Vision_2\runs"
RUN_NAME    = "hardware_fast_v1"

MODEL_SIZE  = "yolov8n.pt"   # NANO — smallest and fastest model
EPOCHS      = 50             # Half the original (early stopping will cut this further)
BATCH_SIZE  = 64             # Large batch = fewer steps per epoch = faster
IMG_SIZE    = 416            # Smaller than default 640 — big speed boost
WORKERS     = 8              # More parallel image loading
# ──────────────────────────────────────────────────────

def train():
    print(f"GPU : {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

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

        # ── SPEED OPTIMIZATIONS ──────────────────────
        cache     = "ram",    # Load entire dataset into RAM — eliminates disk I/O
                              # Needs ~3-4 GB free RAM. Change to False if you run out.
        amp       = True,     # Mixed precision (FP16) — halves VRAM, ~30% faster
        patience  = 15,       # Stop early if no improvement for 15 epochs
        optimizer = "SGD",    # SGD converges faster than AdamW for detection tasks
        
        # ── REDUCED AUGMENTATION (less CPU work per batch) ──
        mosaic    = 0.5,      # Mosaic augmentation 50% of the time (was 1.0)
        mixup     = 0.0,      # Disable mixup (expensive, skip for speed)
        degrees   = 0.0,      # No rotation augmentation
        translate = 0.1,
        scale     = 0.5,
        fliplr    = 0.5,
        flipud    = 0.0,      # No vertical flip
        hsv_h     = 0.015,
        hsv_s     = 0.7,
        hsv_v     = 0.4,

        plots     = True,
        save      = True,
        val       = True,
    )

    best = os.path.join(PROJECT_DIR, RUN_NAME, "weights", "best.pt")
    print(f"\n✓ Done! Best weights saved to:\n  {best}")


if __name__ == "__main__":
    train()
