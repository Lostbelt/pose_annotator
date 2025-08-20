# skelet-annotator

A GUI application for annotating **skeletal keypoints** on images and extracted video frames.  
Supports custom skeletons, **auto-annotation** via Ultralytics YOLO, and export to the **YOLO keypoints format** for training.

<div align="center">
  <img src="docs/screenshot_main.png" width="600"/>
</div>

<div align="center">
  <img src="docs/screenshot_points.png" width="420"/>
</div>

---

## 🚀 Features

- Load an **image folder** or a **video** (frames are extracted automatically).
- Skeleton setup: import from JSON or define _keypoints_ and _connections_ manually.
- Convenient labeling: add/drag keypoints, live table of coordinates, keyboard shortcuts.
- **Auto-annotation** using an Ultralytics YOLO model directly from the GUI.
- **Interpolation** of keypoints between annotated frames.
- Export annotations to:
  - **JSON** (keypoints + bbox computed from user keypoints),
  - **YOLO** (bbox from the model + user keypoints, Ultralytics keypoints format).

---

## 📦 Installation

### Option A. Conda + pip (recommended)

Create and activate an environment:
```bash
conda create -n cv python=3.12 -y
conda activate cv
```

Install core conda deps and all project pip packages (including PySide6 and PyTorch):
```bash
conda install -y numpy pandas scipy matplotlib ipykernel

# GUI + computer vision + utilities
pip install pyside6 shiboken6 ultralytics ultralytics-thop opencv-python pyqtgraph pylsl tqdm sympy requests jinja2 pillow colorama psutil py-cpuinfo PyYAML typing-extensions

# PyTorch (choose the wheel appropriate for your system/driver)
# CUDA example:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
# or CPU-only:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Option B. Using environment files

If the repo contains `environment.yml` and/or `requirements.txt`:
```bash
conda env create -f environment.yml
conda activate cv
# or
pip install -r requirements.txt
```

> **Windows tip:** Install **PySide6 via pip** (not conda) to avoid Qt DLL conflicts.

---

## 📂 Data Preparation

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

## ▶️ Getting Started

Run the GUI:
```bash
python labelboxV3.py
```

Typical workflow:
1. **Choose data**: an image folder or a video to extract frames.
2. **Configure the skeleton**: **File → Setup Skeleton** (manual entry or import from JSON).
3. **(Optional) Load a YOLO model**: left toolbar “open” button → select `.pt` (Ultralytics).
4. Annotate points by clicking; drag to adjust positions.
5. Use **“Interpolate”** to fill in keypoints on in-between frames.
6. Save annotations: **File → Save Annotations As…** (JSON).
7. Export to **YOLO**: **File → Save in YOLO format** (splits into `train/val`).

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
Each `.txt` line includes a bbox (from the model) and keypoints (from user annotations) in Ultralytics format.

---

## 🧩 Example Skeleton JSON

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

---

## 💾 Save Formats

- **JSON** — stores `keypoints`, `connections`, and per-image coordinates. A `bbox` is also computed from user keypoints.
- **YOLO** — uses **bbox from the model** (Ultralytics) and adds **your keypoints** in Ultralytics keypoints format (normalized coordinates + confidence). Files are automatically split into `train/val` sets.

---

## ⛑️ Troubleshooting (Windows)

If you see:
```
qt.qpa.plugin: Could not load the Qt platform plugin "windows" ...
```
this is typically a Qt plugin path conflict. Fixes:

1) Ensure PySide6/shiboken6 are installed **via pip** in the active env.
```powershell
pip install --force-reinstall PySide6 shiboken6
python -c "import PySide6, shiboken6; print(PySide6.__version__)"
```

2) Reset conflicting env vars and point to the platform plugins path:
```powershell
$env:QT_DEBUG_PLUGINS="1"
Remove-Item Env:QT_PLUGIN_PATH -ErrorAction SilentlyContinue
$env:QT_QPA_PLATFORM_PLUGIN_PATH = "$((python -c 'import pathlib,PySide6; print((pathlib.Path(PySide6.__file__).parent / \"plugins\" / \"platforms\").as_posix())'))"
python labelboxV3.py
```

3) Do not mix conda Qt with pip Qt in the same environment.

---

## 📜 License

MIT (or specify your own license).

---

## 🙌 Acknowledgements

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- Qt for Python (PySide6)
