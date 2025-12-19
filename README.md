# Pose annotator

A GUI application for annotating **pose keypoints** on images and extracted video frames.  
Supports custom skeletons, **auto-annotation** via Ultralytics, and export to the **YOLO keypoints format** for training.

<div align="center">
  <img src="figures/main_page.png" width="600"/>
</div>

## Installation


Create an environment:
```bash
conda create -n lab_env Python=3.10
conda activate lab_env

git clone https://github.com/Lostbelt/behaviour_analysis.git
cd behaviour_analysis
pip install -r requirements.txt
# for gpu inference needs cuda PyTorch (choose the wheel appropriate for your system/driver)
# CUDA example:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

> **Windows tip:** Install **PySide6 via pip** (not conda) to avoid Qt DLL conflicts.

---

## Data Preparation

You can either select an **image directory** or a **video file**—the app will offer to extract frames.

Example structure:
```
data/
├── image_0001.jpg
├── image_0002.jpg
├── ...
└── video_1.mp4
```

For auto-annotation, load an Ultralytics YOLO **keypoints** model (`.pt`). See:  
https://docs.ultralytics.com/tasks/keypoints/

---

## Getting Started

Run the GUI:
```bash
python labelboxV3.py
```

Typical workflow:
1. **Choose data**: an image folder or a video to extract frames.
2. **Configure the skeleton**: **File → Setup Skeleton** (manual entry or import from JSON).
3. **(Optional) Load a YOLO model**: left toolbar “open” button → select `.pt` (Ultralytics).
4. Annotate points.
5. Save annotations: **File → Save Annotations As…** (JSON).
6. Export to **YOLO**: **File → Save in YOLO format** (splits into `train/val`).

> Tip: hold `Alt` to draw, move, or resize the bounding box; press `Alt+W` to remove it.

YOLO export structure:
```
dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

## Example Skeleton JSON

```json
{
  "keypoints": [
    "Nose", "LEye", "REye", "LEar", "REar",
    "LShoulder", "RShoulder", "LElbow", "RElbow",
    "LWrist", "RWrist", "LHip", "RHip", "LKnee", "RKnee",
    "LAnkle", "RAnkle"
  ],
  "connections": [
    ["Nose", "LEye"], ["Nose", "REye"],
    ["LEye", "LEar"], ["REye", "REar"],
    ["LShoulder", "RShoulder"],
    ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
    ["RShoulder", "RElbow"], ["RElbow", "RWrist"],
    ["LShoulder", "LHip"], ["RShoulder", "RHip"],
    ["LHip", "RHip"],
    ["LHip", "LKnee"], ["LKnee", "LAnkle"],
    ["RHip", "RKnee"], ["RKnee", "RAnkle"]
  ]
}
```
Load it via **File → Setup Skeleton → Load from JSON**.
