import os
import json
import math
import random
from typing import Dict, List, Tuple, Optional, Callable

import cv2
import numpy as np
import yaml


from PySide6.QtGui import QColor
from PySide6.QtWidgets import QListWidgetItem


QColor = None
QListWidgetItem = object


# ============================================================
# I/O and files
# ============================================================

def safe_imread(path: str) -> Optional[np.ndarray]:
    """
    Safe image loader (cv2.imread returns None on failure).
    """
    if not path or not os.path.exists(path):
        return None
    img = cv2.imread(path)
    return img


def list_images(directory: str,
                exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"),
                try_frame_sort: bool = True) -> List[str]:
    """
    Return a sorted list of image paths. When try_frame_sort=True, attempts to sort by frame index (frame_###.ext).
    """
    files = [os.path.join(directory, f)
             for f in os.listdir(directory)
             if os.path.splitext(f)[1].lower() in exts]
    if try_frame_sort:
        try:
            files.sort(key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
            return files
        except Exception:
            pass
    files.sort()
    return files


def load_annotations(path: str) -> Dict:
    """
    Load annotations JSON. Returns a dict (or empty dict) with the structure:
    {
      "keypoints": [...],
      "connections": [...],
      "annotations": { "img_path": {"points": {...}, "bbox": [x1,y1,x2,y2]}, ... }
    }
    """
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_annotations(path: str, data: Dict, indent: int = 4) -> bool:
    """
    Save annotations JSON. Returns True on success.
    """
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        return True
    except Exception:
        return False


def extract_frames_from_video(video_path: str,
                              mode: str = "step",  # "step" | "sequential" | "all"
                              count: int = 10,
                              out_dir: Optional[str] = None) -> List[str]:
    """
    Extract frames from video and return saved frame paths.
    mode="step": evenly spaced across the video
    mode="sequential": first `count` frames in sequence
    mode="all": every frame
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    if out_dir is None:
        base = os.path.splitext(os.path.basename(video_path))[0]
        out_dir = os.path.join(os.getcwd(), f"{base}_frames")
    os.makedirs(out_dir, exist_ok=True)

    if mode == "all":
        indices = list(range(total))
    elif mode == "sequential":
        indices = list(range(min(count, total)))
    else:  # step
        step = max(total // max(1, count), 1)
        indices = [i * step for i in range(min(count, math.ceil(total / step)))]

    paths = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        out_path = os.path.join(out_dir, f"frame_{idx}.jpg")
        cv2.imwrite(out_path, frame)
        paths.append(out_path)
    cap.release()
    return paths


# ============================================================
# Geometry / transforms
# ============================================================


def xywh_to_xyxy(cx: float, cy: float, w: float, h: float) -> Tuple[float, float, float, float]:
    """
    Convert center format to xyxy in absolute pixels (assuming cx,cy,w,h already in pixels).
    """
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return x1, y1, x2, y2


def abs_to_norm(x: float, y: float, W: int, H: int) -> Tuple[float, float]:
    """
    Absolute pixels -> normalized coordinates (0..1).
    """
    return x / max(1, W), y / max(1, H)



def clamp_bbox_to_image(bbox_xyxy: List[float], W: int, H: int) -> List[int]:
    """
    Clamp bbox to image bounds, returning integer [x1,y1,x2,y2].
    """
    x1, y1, x2, y2 = bbox_xyxy
    x1 = int(max(0, min(W - 1, x1)))
    y1 = int(max(0, min(H - 1, y1)))
    x2 = int(max(0, min(W - 1, x2)))
    y2 = int(max(0, min(H - 1, y2)))
    return [x1, y1, x2, y2]


def bbox_from_points(points_dict: Dict[str, Tuple[float, float]]) -> Optional[List[int]]:
    """
    Build a bbox [x1,y1,x2,y2] from a dict of keypoints (if any).
    """
    if not points_dict:
        return None
    xs = [c[0] for c in points_dict.values() if c is not None]
    ys = [c[1] for c in points_dict.values() if c is not None]
    if not xs or not ys:
        return None
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]


# ============================================================
# Interpolation
# ============================================================

def interpolate_points_between_frames(points_start: Dict[str, Tuple[float, float]],
                                      points_end: Dict[str, Tuple[float, float]],
                                      t: float) -> Dict[str, Tuple[float, float]]:
    """
    Linear interpolation of points for t ∈ [0,1], using only shared keys.
    """
    out: Dict[str, Tuple[float, float]] = {}
    for kp, p1 in points_start.items():
        if kp in points_end and p1 is not None and points_end[kp] is not None:
            x1, y1 = p1
            x2, y2 = points_end[kp]
            x = x1 + (x2 - x1) * t
            y = y1 + (y2 - y1) * t
            out[kp] = (x, y)
    return out


def interpolate_range(image_paths: List[str],
                      annotations: Dict[str, Dict],
                      keypoints_order: List[str],
                      start_idx: int,
                      end_idx: int) -> None:
    """
    Interpolate intermediate frames between start_idx and end_idx (excluding endpoints).
    Mutates `annotations` in place.
    """
    p_start = image_paths[start_idx]
    p_end = image_paths[end_idx]
    pts_start = annotations.get(p_start, {}).get("points", {})
    pts_end = annotations.get(p_end, {}).get("points", {})

    total = end_idx - start_idx
    if total < 2:
        return

    for mid in range(start_idx + 1, end_idx):
        t = (mid - start_idx) / total
        mid_path = image_paths[mid]
        if mid_path not in annotations:
            annotations[mid_path] = {"points": {}}
        inter = interpolate_points_between_frames(pts_start, pts_end, t)
        annotations[mid_path]["points"].update(inter)


# ============================================================
# Navigation / helpers
# ============================================================

def is_frame_annotated(annotations: Dict[str, Dict], path: str) -> bool:
    """
    Treat a frame as annotated if it has at least one point or a bbox.
    """
    rec = annotations.get(path, {})
    return bool(rec.get("points")) or bool(rec.get("bbox"))


def find_next_matching_index(items: List[str],
                             start_index: int,
                             predicate: Callable[[str], bool]) -> Optional[int]:
    """
    Cyclically search for the next index that satisfies predicate.
    Returns None if nothing matches.
    """
    n = len(items)
    if n == 0:
        return None
    start = start_index + 1
    for k in range(n):
        i = (start + k) % n
        if predicate(items[i]):
            return i
    return None


def colorize_list_item(item, annotated: bool) -> None:
    """
    Colorize a list item (if QColor is available).
    """
    if QColor is None or item is None:
        return
    if annotated:
        item.setForeground(QColor("#1b5e20"))
        item.setBackground(QColor("#c8e6c9"))
    else:
        item.setForeground(QColor("#616161"))
        item.setBackground(QColor("#f5f5f5"))


# ============================================================
# YOLO helpers (Ultralytics)
# ============================================================

def yolo_infer(model, image: np.ndarray, conf: float = 0.5):
    """
    Run an Ultralytics YOLO model and return the first result (res[0]).
    """
    if model is None or image is None:
        return None
    res = model(image, verbose=False, conf=conf)
    return res[0] if res and len(res) > 0 else None


def yolo_best_bbox(res, img_shape: Tuple[int, int]) -> Optional[List[int]]:
    """
    Take the most confident bbox from YOLO output.
    Returns absolute-pixel [x1,y1,x2,y2] or None.
    """
    if res is None or not hasattr(res, "boxes"):
        return None
    H, W = img_shape
    try:
        boxes_n = res.boxes.xywhn.to('cpu').numpy()  # (N,4)
        confs = res.boxes.conf.to('cpu').numpy() if hasattr(res.boxes, 'conf') else None
    except Exception:
        return None

    if boxes_n is None or boxes_n.shape[0] == 0:
        return None

    idx = int(np.argmax(confs)) if confs is not None and len(confs) == boxes_n.shape[0] else 0
    cx_n, cy_n, w_n, h_n = boxes_n[idx]
    cx, cy = cx_n * W, cy_n * H
    w, h = w_n * W, h_n * H
    x1, y1, x2, y2 = xywh_to_xyxy(cx, cy, w, h)
    return clamp_bbox_to_image([x1, y1, x2, y2], W, H)


def yolo_keypoints(res, keypoints_order: List[str]) -> Dict[str, Tuple[float, float]]:
    """
    Extract keypoints from YOLO output and map them to keypoints_order.
    Skips (0,0) pairs.
    """
    out: Dict[str, Tuple[float, float]] = {}
    if res is None or not hasattr(res, "keypoints") or not hasattr(res.keypoints, "xy"):
        return out
    try:
        raw = res.keypoints.xy.to('cpu').numpy()
    except Exception:
        return out
    if raw.ndim == 3:  # (N, K, 2)
        raw = raw[0]
    for i, name in enumerate(keypoints_order):
        if i >= raw.shape[0]:
            break
        x, y = raw[i][:2]
        if float(x) == 0.0 and float(y) == 0.0:
            continue
        out[name] = (float(x), float(y))
    return out


def annotate_points_and_bbox(model,
                             image: np.ndarray,
                             keypoints_order: List[str],
                             conf: float = 0.5) -> Dict[str, Optional[Dict]]:
    """
    Convenience wrapper: runs YOLO and returns {"points": {...}, "bbox": [...] or None}.
    """
    res = yolo_infer(model, image, conf)
    if res is None:
        return {"points": {}, "bbox": None}
    pts = yolo_keypoints(res, keypoints_order)
    bbox = yolo_best_bbox(res, image.shape[:2])
    return {"points": pts, "bbox": bbox}


# ============================================================
# Export helpers
# ============================================================

def make_yolo_line(img_shape: Tuple[int, int],
                   bbox_xyxy: Optional[List[int]],
                   keypoints_order: List[str],
                   points_dict: Dict[str, Tuple[float, float]],
                   cls_id: int = 0) -> Optional[str]:
    """
    Build a YOLO keypoints line: 'cls cx cy w h (x y v)*'.
    - bbox is absolute pixels [x1,y1,x2,y2]; if None, returns None.
    - keypoints are normalized to [0..1], visibility=1.0, missing ones are 0 0 0.
    """
    H, W = img_shape
    if bbox_xyxy is None:
        return None
    x1, y1, x2, y2 = bbox_xyxy
    cx = (x1 + x2) / 2 / max(1, W)
    cy = (y1 + y2) / 2 / max(1, H)
    bw = abs(x2 - x1) / max(1, W)
    bh = abs(y2 - y1) / max(1, H)

    parts = [str(int(cls_id)), f"{cx:.6f}", f"{cy:.6f}", f"{bw:.6f}", f"{bh:.6f}"]
    for kp in keypoints_order:
        if kp in points_dict and points_dict[kp] is not None:
            x, y = points_dict[kp]
            xn, yn = abs_to_norm(float(x), float(y), W, H)
            parts += [f"{xn:.6f}", f"{yn:.6f}", "1.0"]
        else:
            parts += ["0.000000", "0.000000", "0.0"]
    return " ".join(parts)


def train_val_split(paths: List[str], ratio: float = 0.8, seed: Optional[int] = None) -> Tuple[List[str], List[str]]:
    """
    Split paths into train/val using probability `ratio`.
    """
    if seed is not None:
        random.seed(seed)
    train, val = [], []
    for p in paths:
        (train if random.random() < max(0.0, min(1.0, ratio)) else val).append(p)
    return train, val


def write_yolo_dataset_yaml(out_dir: str,
                            keypoints: List[str],
                            connections: List[Tuple[str, str]],
                            class_name: str = "object") -> str:
    """
    Create an Ultralytics YOLO pose YAML config and return its path.
    """
    os.makedirs(out_dir, exist_ok=True)
    kp_to_idx = {name: idx for idx, name in enumerate(keypoints)}
    skeleton = []
    for a, b in connections:
        if a in kp_to_idx and b in kp_to_idx:
            skeleton.append([kp_to_idx[a], kp_to_idx[b]])

    yaml_path = os.path.join(out_dir, "dataset.yaml")
    kp_text = yaml.safe_dump(
        keypoints,
        default_flow_style=True,
        allow_unicode=True,
        sort_keys=False
    ).strip()
    skeleton_text = None
    if skeleton:
        skeleton_text = yaml.safe_dump(
            skeleton,
            default_flow_style=True,
            allow_unicode=True,
            sort_keys=False
        ).strip()

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {os.path.abspath(out_dir)}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n\n")
        f.write("# Keypoints\n")
        f.write(f"kpt_shape: [{len(keypoints)}, 3]  # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)\n")
        f.write(f"keypoints: {kp_text}\n\n")
        if skeleton_text:
            f.write(f"skeleton: {skeleton_text}\n\n")
        f.write("# Classes\n")
        f.write(f"names: ['{class_name}']\n")
    return yaml_path


# ============================================================
# Validation / QC
# ============================================================


def ensure_bbox_present(record: Dict) -> None:
    """
    If a record lacks bbox but has points, derive bbox from them (mutates record).
    """
    if not isinstance(record, dict):
        return
    if record.get("bbox"):
        return
    pts = record.get("points", {}) or {}
    bb = bbox_from_points(pts)
    if bb is not None:
        record["bbox"] = bb


# ============================================================
# Snapshots (Undo/Redo)
# ============================================================

def snapshot_annotation(annotations: Dict[str, Dict], path: str) -> Tuple[str, Dict]:
    """
    Return a deep copy of annotation record for path.
    """
    rec = annotations.get(path, {"points": {}, "bbox": None})
    return path, json.loads(json.dumps(rec))  # deep copy via JSON


def restore_snapshot(annotations: Dict[str, Dict], path: str, snapshot: Dict) -> None:
    """
    Restore a snapshot for the given path.
    """
    annotations[path] = json.loads(json.dumps(snapshot))
