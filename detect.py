"""
YOLOv8 Hardware Object Detection — Inference Script
Supports: single image, folder of images, webcam
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from ultralytics import YOLO

# ─── CONFIG ───────────────────────────────────────────────────
WEIGHTS   = r"C:\Users\Arjun\Vision_2\runs\hardware_s_v1\weights\best.pt"

# ── Set your source here ──
# Single image  : r"C:\path\to\image.jpg"
# Folder        : r"C:\path\to\folder"
# Webcam        : 0
SOURCE    = r"C:\path\to\your\image_or_folder"

CONF      = 0.25       # Confidence threshold (0-1). Raise to reduce false detections.
IOU       = 0.45       # Overlap threshold for removing duplicate boxes
IMG_SIZE  = 640        # Must match what you trained with
SAVE_DIR  = r"C:\Users\Arjun\Vision_2\results"   # Where annotated images are saved
# ──────────────────────────────────────────────────────────────


def run():
    # Verify weights exist
    if not os.path.exists(WEIGHTS):
        raise SystemExit(f"\nERROR: Weights not found at:\n  {WEIGHTS}")

    # Verify source exists
    if SOURCE != 0 and not os.path.exists(SOURCE):
        raise SystemExit(f"\nERROR: Source not found at:\n  {SOURCE}")

    print(f"\n  Weights : {WEIGHTS}")
    print(f"  Source  : {SOURCE}")
    print(f"  Conf    : {CONF}")
    print(f"  Output  : {SAVE_DIR}\n")

    model = YOLO(WEIGHTS)

    results = model.predict(
        source     = SOURCE,
        conf       = CONF,
        iou        = IOU,
        imgsz      = IMG_SIZE,
        device     = 0 if torch.cuda.is_available() else "cpu",
        save       = True,         # Save annotated images
        save_txt   = True,         # Save detections as .txt label files
        save_conf  = True,         # Include confidence scores in .txt files
        project    = SAVE_DIR,
        name       = "detections",
        exist_ok   = True,
        line_width = 2,            # Bounding box line thickness
        show       = False,        # Set True to pop up a window showing each result
    )

    # ── Print summary ──────────────────────────────────────────
    print("\n" + "=" * 50)
    total_detections = 0
    for r in results:
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            print(f"  {os.path.basename(str(r.path))} — no detections")
            continue

        # Count per class
        class_counts = {}
        for cls_id, conf in zip(boxes.cls.tolist(), boxes.conf.tolist()):
            name = model.names[int(cls_id)]
            class_counts[name] = class_counts.get(name, 0) + 1
            total_detections += 1

        summary = ", ".join([f"{v}x {k}" for k, v in class_counts.items()])
        print(f"  {os.path.basename(str(r.path))} → {summary}")

    print("=" * 50)
    print(f"\n  Total detections : {total_detections}")
    print(f"  Annotated images saved to:")
    print(f"  {SAVE_DIR}\\detections\\\n")


if __name__ == "__main__":
    run()
