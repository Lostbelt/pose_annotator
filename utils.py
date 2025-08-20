import os
import json
import math
import random
from typing import Dict, List, Tuple, Optional, Callable

import cv2
import numpy as np

# Опциональные GUI-типы (для окраски элементов списка)
try:
    from PySide6.QtGui import QColor
    from PySide6.QtWidgets import QListWidgetItem
except Exception:  # PySide6 может отсутствовать при headless-скриптах
    QColor = None
    QListWidgetItem = object  # type: ignore


# ============================================================
# I/O и файлы
# ============================================================

def safe_imread(path: str) -> Optional[np.ndarray]:
    """
    Безопасное чтение изображения (cv2.imread возвращает None при неудаче).
    """
    if not path or not os.path.exists(path):
        return None
    img = cv2.imread(path)
    return img


def list_images(directory: str,
                exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"),
                try_frame_sort: bool = True) -> List[str]:
    """
    Список изображений в каталоге. При try_frame_sort пытается сортировать по номеру кадра вида frame_###.ext
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
    Загрузка аннотаций JSON. Возвращает dict (или пустой dict).
    Ожидаемый формат верхнего уровня:
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
    Сохранение аннотаций JSON. Возвращает True при успехе.
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
    Извлекает кадры из видео. Возвращает список путей сохранённых кадров.
    mode="step": равномерный шаг по всему видео
    mode="sequential": первые `count` кадров подряд
    mode="all": все кадры
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
# Геометрия / преобразования
# ============================================================

def xyxy_to_xywh(x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float, float, float]:
    """
    Преобразование [x1,y1,x2,y2] -> [x,y,w,h]
    """
    x = min(x1, x2)
    y = min(y1, y2)
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    return x, y, w, h


def xywh_to_xyxy(cx: float, cy: float, w: float, h: float) -> Tuple[float, float, float, float]:
    """
    Преобразование центр-нормы -> xyxy в абсолютных пикселях (если данные абсолютные).
    Здесь предполагаем, что cx,cy,w,h — уже в пикселях. (Используйте abs_to_norm/norm_to_abs при необходимости.)
    """
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return x1, y1, x2, y2


def abs_to_norm(x: float, y: float, W: int, H: int) -> Tuple[float, float]:
    """
    Абсолютные пиксели -> нормированные координаты (0..1)
    """
    return x / max(1, W), y / max(1, H)


def norm_to_abs(xn: float, yn: float, W: int, H: int) -> Tuple[float, float]:
    """
    Нормированные координаты (0..1) -> абсолютные пиксели
    """
    return xn * W, yn * H


def clamp_bbox_to_image(bbox_xyxy: List[float], W: int, H: int) -> List[int]:
    """
    Ограничивает bbox границами изображения. Возвращает целочисленные [x1,y1,x2,y2].
    """
    x1, y1, x2, y2 = bbox_xyxy
    x1 = int(max(0, min(W - 1, x1)))
    y1 = int(max(0, min(H - 1, y1)))
    x2 = int(max(0, min(W - 1, x2)))
    y2 = int(max(0, min(H - 1, y2)))
    return [x1, y1, x2, y2]


def bbox_from_points(points_dict: Dict[str, Tuple[float, float]]) -> Optional[List[int]]:
    """
    Строит bbox [x1,y1,x2,y2] по словарю ключевых точек (если есть хотя бы одна точка).
    """
    if not points_dict:
        return None
    xs = [c[0] for c in points_dict.values() if c is not None]
    ys = [c[1] for c in points_dict.values() if c is not None]
    if not xs or not ys:
        return None
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]


# ============================================================
# Интерполяция
# ============================================================

def interpolate_points_between_frames(points_start: Dict[str, Tuple[float, float]],
                                      points_end: Dict[str, Tuple[float, float]],
                                      t: float) -> Dict[str, Tuple[float, float]]:
    """
    Линейная интерполяция точек, где t ∈ [0,1].
    Берутся только общие точки, присутствующие в обоих словарях.
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
    Интерполирует промежуточные кадры между start_idx и end_idx (исключая сами границы).
    Модифицирует `annotations` на месте.
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
# Навигация / сервис
# ============================================================

def is_frame_annotated(annotations: Dict[str, Dict], path: str) -> bool:
    """
    Считаем кадр размеченным, если есть хотя бы одна точка или bbox.
    """
    rec = annotations.get(path, {})
    return bool(rec.get("points")) or bool(rec.get("bbox"))


def find_next_matching_index(items: List[str],
                             start_index: int,
                             predicate: Callable[[str], bool]) -> Optional[int]:
    """
    Циклический поиск следующего индекса, удовлетворяющего предикату.
    Возвращает None, если ничего не найдено.
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
    Окраска элемента списка (если доступен QColor).
    """
    if QColor is None or item is None:
        return
    item.setForeground(QColor(0, 200, 120) if annotated else QColor(200, 200, 200))


# ============================================================
# Работа с YOLO (Ultralytics)
# ============================================================

def yolo_infer(model, image: np.ndarray, conf: float = 0.5):
    """
    Запускает модель Ultralytics YOLO и возвращает первый результат (res[0]).
    """
    if model is None or image is None:
        return None
    res = model(image, verbose=False, conf=conf)
    return res[0] if res and len(res) > 0 else None


def yolo_best_bbox(res, img_shape: Tuple[int, int]) -> Optional[List[int]]:
    """
    Извлекает самый уверенный bbox из результата YOLO.
    Возвращает bbox в абсолютных пикселях [x1,y1,x2,y2] либо None.
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
    Извлекает keypoints из результата YOLO и сопоставляет их именам из keypoints_order.
    Пропускает пары (0,0).
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
    Комбайн: запускает YOLO, возвращает {"points": {...}, "bbox": [x1,y1,x2,y2] или None}
    """
    res = yolo_infer(model, image, conf)
    if res is None:
        return {"points": {}, "bbox": None}
    pts = yolo_keypoints(res, keypoints_order)
    bbox = yolo_best_bbox(res, image.shape[:2])
    return {"points": pts, "bbox": bbox}


# ============================================================
# Экспорт форматов
# ============================================================

def make_yolo_line(img_shape: Tuple[int, int],
                   bbox_xyxy: Optional[List[int]],
                   keypoints_order: List[str],
                   points_dict: Dict[str, Tuple[float, float]],
                   cls_id: int = 0) -> Optional[str]:
    """
    Формирует строку YOLO (keypoints): 'cls cx cy w h (x y v)*'
    - bbox задаётся в абсолютных пикселях [x1,y1,x2,y2]; если None — вернёт None.
    - keypoints нормируются в [0..1], visibility=1.0, а для отсутствующих — 0 0 0.
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
    Делит пути на train/val по вероятности ratio.
    """
    if seed is not None:
        random.seed(seed)
    train, val = [], []
    for p in paths:
        (train if random.random() < max(0.0, min(1.0, ratio)) else val).append(p)
    return train, val


def build_coco_json(image_paths: List[str],
                    annotations: Dict[str, Dict],
                    keypoints: List[str],
                    connections: List[Tuple[str, str]]) -> Dict:
    """
    Формирует COCO JSON-объект (images, annotations, categories) для позы (2D keypoints).
    """
    images = []
    ann_list = []
    categories = [{
        "id": 1,
        "name": "object",
        "supercategory": "object",
        "keypoints": keypoints,
        "skeleton": [[keypoints.index(a) + 1, keypoints.index(b) + 1] for a, b in connections]
    }]

    ann_id = 1
    for img_id, path in enumerate(image_paths, start=1):
        img = safe_imread(path)
        if img is None:
            continue
        H, W = img.shape[:2]
        images.append({"id": img_id, "file_name": os.path.basename(path), "width": W, "height": H})

        rec = annotations.get(path, {})
        pts = rec.get("points", {}) or {}
        bbox = rec.get("bbox")

        # keypoints flat [x,y,v] * K
        kp_flat = []
        num_kp = 0
        for kp in keypoints:
            if kp in pts and pts[kp] is not None:
                x, y = map(float, pts[kp])
                kp_flat += [x, y, 2]  # v=2 размечен/видим
                num_kp += 1
            else:
                kp_flat += [0.0, 0.0, 0]

        if bbox is None:
            bbox = bbox_from_points(pts)
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            x, y, w, h = xyxy_to_xywh(x1, y1, x2, y2)
        else:
            x, y, w, h = 0.0, 0.0, 0.0, 0.0

        if num_kp > 0 or (w * h) > 0:
            ann_list.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "iscrowd": 0,
                "keypoints": kp_flat,
                "num_keypoints": num_kp,
                "bbox": [x, y, w, h],
                "area": w * h
            })
            ann_id += 1

    return {"images": images, "annotations": ann_list, "categories": categories}


def write_openpose_jsons(image_paths: List[str],
                         annotations: Dict[str, Dict],
                         keypoints: List[str],
                         out_dir: str) -> List[str]:
    """
    Записывает OpenPose style JSON для каждого изображения.
    Возвращает список путей сохранённых файлов.
    """
    os.makedirs(out_dir, exist_ok=True)
    saved = []
    for path in image_paths:
        rec = annotations.get(path, {})
        pts = rec.get("points", {}) or {}
        flat = []
        for kp in keypoints:
            if kp in pts and pts[kp] is not None:
                x, y = map(float, pts[kp])
                flat += [x, y, 1.0]
            else:
                flat += [0.0, 0.0, 0.0]
        data = {"version": 1.3, "people": [{"person_id": [-1], "pose_keypoints_2d": flat}]}
        fname = os.path.splitext(os.path.basename(path))[0] + "_keypoints.json"
        out_path = os.path.join(out_dir, fname)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        saved.append(out_path)
    return saved


# ============================================================
# Валидация / QC
# ============================================================

def validate_skeleton(keypoints: List[str], connections: List[Tuple[str, str]]) -> bool:
    """
    Проверяет, что все связи указывают на существующие keypoints.
    """
    ks = set(keypoints)
    for a, b in connections:
        if a not in ks or b not in ks:
            return False
    return True


def validate_annotation_record(record: Dict, W: int, H: int) -> bool:
    """
    Простая проверка, что точки и bbox лежат в пределах изображения.
    Не падает на None/пустых значениях.
    """
    if not isinstance(record, dict):
        return False

    pts = record.get("points", {})
    if isinstance(pts, dict):
        for name, v in pts.items():
            if v is None:
                continue
            x, y = v
            if not (0 <= float(x) < W and 0 <= float(y) < H):
                return False

    bbox = record.get("bbox")
    if bbox:
        x1, y1, x2, y2 = bbox
        if not (0 <= x1 < W and 0 <= x2 < W and 0 <= y1 < H and 0 <= y2 < H):
            return False
    return True


def ensure_bbox_present(record: Dict) -> None:
    """
    Если в записи нет bbox, но есть точки — добавляет bbox по точкам.
    Модифицирует record на месте.
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
# Снапшоты (для Undo/Redo)
# ============================================================

def snapshot_annotation(annotations: Dict[str, Dict], path: str) -> Tuple[str, Dict]:
    """
    Возвращает глубокую копию записи аннотаций для path.
    """
    rec = annotations.get(path, {"points": {}, "bbox": None})
    return path, json.loads(json.dumps(rec))  # глуб. копия через JSON


def restore_snapshot(annotations: Dict[str, Dict], path: str, snapshot: Dict) -> None:
    """
    Восстанавливает снапшот записи по path.
    """
    annotations[path] = json.loads(json.dumps(snapshot))