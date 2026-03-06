# Hardware Object Detection — YOLOv8

Real-time detection of hardware components (screws, bolts, nuts, washers) using YOLOv8.

---

## Results

### YOLOv8s (Current Best Model)
| Metric | Value |
|--------|-------|
| Training Time | 5 hours 20 minutes |
| Epochs | 100 |
| Image Size | 640×640 |
| mAP50 | **0.732** |
| mAP50-95 | **0.500** |

#### Per-Class Performance
| Class | Precision | Recall | mAP50 | mAP50-95 |
|-------|-----------|--------|-------|----------|
| Bolt | 0.875 | 0.806 | 0.862 | 0.695 |
| Screw | 0.797 | 0.750 | 0.792 | 0.483 |
| Nut | 0.789 | 0.602 | 0.669 | 0.480 |
| Washer | 0.821 | 0.503 | 0.605 | 0.340 |
| **All** | **0.820** | **0.665** | **0.732** | **0.500** |

### YOLOv8n (Baseline)
| Metric | Value |
|--------|-------|
| Training Time | ~50 minutes |
| Epochs | 50 |
| Image Size | 416×416 |
| mAP50 | 0.672 |
| mAP50-95 | 0.422 |

---

## Dataset
Dataset sourced from Roboflow Universe:
**[ARG Bolts FV — Roboflow](https://app.roboflow.com/vision05/arg_bolts_fv-ojjwp/1)**

- 9008 training images
- 2563 validation images
- 4 classes: Bolt, Screw, Nut, Washer
- Format: YOLOv8 (bounding box + polygon labels)

---

## Project Structure
```
Vision_2/
├── train_yolov8.py           # Full training script (accuracy-focused)
├── train_yolov8_fast.py      # Fast training script (speed-optimized)
├── detect.py                 # Inference script — single image, folder, or webcam
└── Data/
    ├── train/                # 9008 training images + labels
    ├── valid/                # 2563 validation images + labels
    ├── test/                 # Test images + labels
    └── data.yaml             # Dataset config
```

---

## Setup & Usage

### 1. Install dependencies
```bash
pip install ultralytics torch torchvision
```

### 2. Download the dataset
Get the dataset from [Roboflow](https://app.roboflow.com/vision05/arg_bolts_fv-ojjwp/1) and place it in the `Data/` folder.

### 3. Train
```bash
# Fast training — yolov8s, 640px (recommended)
python train_yolov8_fast.py

# Full training — yolov8m, 640px (more accurate, slower)
python train_yolov8.py
```

### 4. Run detection on your own images
Edit the `SOURCE` line in `detect.py`:
```python
SOURCE = r"C:\path\to\your\image.jpg"   # single image
SOURCE = r"C:\path\to\your\folder"      # folder of images
SOURCE = 0                              # webcam
```
Then run:
```bash
python detect.py
```
Annotated results are saved to `results/detections/`.

---

## Hardware & Environment
| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4050 Laptop (6GB VRAM) |
| Python | 3.11.7 |
| PyTorch | 2.5.1 + CUDA 12.1 |
| Ultralytics | 8.4.12 |

---

## License
MIT
