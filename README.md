# Pose annotator

A GUI application for annotating **pose keypoints** on images and extracted video frames.  
Supports custom skeletons, **auto-annotation** via Ultralytics, and export to the **YOLO keypoints format** for training.

<div align="center">
  <img src="figures/main_page.png" width="600"/>
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

Create an environment using yml file:
```bash
conda env create -f environment.yml
conda activate cv
# for gpu inference needs cuda PyTorch (choose the wheel appropriate for your system/driver)
# CUDA example:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
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
The tool also writes a ready-to-use `dataset.yaml` inside the export folder (Ultralytics format).

---

## 🧭 Detailed Instructions

### 1. Launch & Load Data
- Start the app with `python labelboxV3.py`.
- The initial dialog offers two options:
  1. **Image folder** – choose a directory with still frames.
  2. **Video file** – pick a video; the app lets you extract frames (step, sequential, or all).
- The left dock lists every frame. Use `A` / `D` keys or the “Previous/Next” buttons to navigate.

### 2. Configure the Skeleton
- Open **File → Configure Skeleton**.
- Either paste keypoint names + connections or import a JSON (see sample below).
- After saving, the point table on the right shows all keypoints; blank cells indicate unplaced points.

### 3. Annotating Frames
- Click anywhere on the image to place the currently selected keypoint. The tool automatically selects the next unfilled point.
- Drag an existing point to move it. Press `W` to remove the point under the cursor.
- The right dock displays:
  - **BBox** table (single-row summary).
  - **Keypoint** table (click a row to jump to that point).
- Frame list coloring:
  - **Green background** – frame has at least one annotation.
  - **Gray background** – empty frame.

### 4. Bounding Boxes
- Hold **`Alt`** to interact with the bbox:
  - Alt + drag to draw a new box.
  - Alt + drag edges to resize / move.
  - **Alt + W** deletes the current box.
- Outside of Alt mode you can’t interact with the bbox, so regular point placement stays safe.

### 5. Auto Annotation
- Load an Ultralytics keypoints model via the toolbar button (choose device & confidence on the left panel).
- Enable “Auto on select” to infer missing frames automatically while browsing.
- Use “Auto annotate all” to process every frame; progress appears in a modal dialog.

### 6. Interpolation & Cleanup
- Use **Interpolate** when at least two frames are annotated; the app fills the in-between frames.
- “Clear all” wipes current annotations (undo is available).

### 7. Saving & Exporting
- Save annotations anytime via **File → Save annotations as…**; autosave runs every minute.
- Export to YOLO via **File → Save as YOLO**:
  - Choose train/val split ratio.
  - Images & labels are copied into the standard Ultralytics directory tree.
  - `dataset.yaml` is created in the chosen folder with the exact keypoints and skeleton from your annotations.

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
