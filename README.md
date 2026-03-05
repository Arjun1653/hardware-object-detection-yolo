# Hardware Object Detection — YOLOv8

Real-time detection of hardware components (screws, bolts, nuts, washers) using YOLOv8.

---

## Demo
> Training in progress — results will be added here after training completes.

---

## Project Structure
```
Vision_2/
├── train_yolov8.py          # Full training script (accuracy-focused)
├── train_yolov8_fast.py     # Fast training script (speed-optimized)
└── Data/
    ├── train/               # 9008 training images + labels
    ├── valid/               # 2563 validation images + labels
    ├── test/                # Test images + labels
    └── data.yaml            # Dataset config
```

---

## Dataset
Dataset sourced from Roboflow Universe:
**[ARG Bolts FV — Roboflow](https://app.roboflow.com/vision05/arg_bolts_fv-ojjwp/1)**

- 9008 training images
- 2563 validation images
- 4 classes: screws, bolts, nuts, washers
- Format: YOLOv8 (bounding box + polygon labels)

---

## Model
- Architecture: YOLOv8 (Ultralytics)
- Variant used: `yolov8n` (nano) for fast training, `yolov8m` (medium) for accuracy
- Input resolution: 416×416
- GPU: NVIDIA RTX 4050 Laptop (6GB VRAM)

---

## Setup & Usage

### 1. Install dependencies
```bash
pip install ultralytics torch torchvision
```

### 2. Download the dataset
Get the dataset from [Roboflow](https://app.roboflow.com/vision05/arg_bolts_fv-ojjwp/1) and place it in the `Data/` folder.

### 3. Run training
```bash
# Fast training (recommended to start)
python train_yolov8_fast.py

# Full training (more accurate, slower)
python train_yolov8.py
```

### 4. Results
Trained weights and plots are saved to:
```
runs/hardware_fast_v2/weights/best.pt
```

---

## Requirements
- Python 3.11
- PyTorch 2.5+ with CUDA 12.1
- Ultralytics 8.4+
- NVIDIA GPU (CUDA-capable)

---

## License
MIT