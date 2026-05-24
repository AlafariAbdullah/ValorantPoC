# ValorantPoC 🎮

A computer vision proof of concept for analyzing Valorant's minimap, built for the **Twisted Minds Esports Hackathon 2025**.

## Overview

This project uses **YOLOv8** to detect and track players, enemies, and game objects on the Valorant minimap in real time. It identifies when a friendly player becomes **isolated from their teammates** — a key indicator of poor coordination or map control — and highlights them visually.

Built and demoed on a **MacBook Air (M1)** using PyTorch's MPS backend at ~15–20 FPS. Developed for the **Twisted Minds Esports Hackathon 2025** in Riyadh.

> Read the full writeup: [Blog Post](https://alafariabdullah.sa/detecting-player-isolation-in-valorant-using-yolov8/)
## Files

- **`minimalcam.py`** — Main script (~300 lines). Captures the minimap in real time, runs YOLOv8 inference, tracks players with a custom IOU tracker, and highlights isolated teammates.
- **`train_yolo.py`** — Trains a YOLOv8n model on the custom Valorant minimap dataset.
- **`StopRotation.py`** — Experimental script that analyzes Valorant gameplay video to detect and predict player rotations using OpenCV.
- **`test.py`** — Uses SAM2 (Segment Anything Model 2) to auto-generate segmentation masks on a Valorant map image.
- **`automatic_mask_generator_example.ipynb`** — Jupyter notebook demonstrating SAM2 automatic mask generation.

## Requirements

- Python 3.10+
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [SAM2](https://github.com/facebookresearch/segment-anything-2)
- OpenCV (`cv2`)
- PyTorch
- `mss` (screen capture)

Install dependencies:
```bash
pip install ultralytics opencv-python torch mss
```

> **Note:** This project was developed using Apple's Metal (`mps`) backend. For NVIDIA GPUs use `cuda:0`, or `cpu` for general compatibility.

## Usage

**Run live minimap detection:**
```bash
python minimalcam.py
```

**Train the YOLO model:**
```bash
python train_yolo.py
```

**Analyze rotation from gameplay video:**
```bash
python StopRotation.py
```

## Dataset

Trained on the [Roboflow Valorant Map Analyzer](https://universe.roboflow.com/valomap/valorant-map-analyser-r3kxb/browse) dataset (40+ classes), remapped into three categories:

- 🟩 **Friendly**
- 🟥 **Enemy**
- 🟦 **Other / Neutral** (Spike, last seen, dead players)

## Technical Summary

| Field | Details |
|-------|---------|
| **Language** | Python |
| **Frameworks** | Ultralytics YOLOv8, OpenCV, PyTorch, MSS |
| **Script** | `minimalcam.py` (≈300 lines) |
| **Hardware** | MacBook Air M1, MPS backend |
| **Performance** | ~15–20 FPS at 640×640 |

## Hackathon

Built for the **Twisted Minds Esports Hackathon 2025** in Riyadh.
