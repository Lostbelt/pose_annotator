import os
import sys
import json
import cv2
import torch
import shutil
from typing import List, Tuple, Optional, Dict

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFileDialog, QDialog, QMessageBox, QComboBox, QMenuBar, QGraphicsScene,
    QGraphicsView, QDockWidget, QListWidget, QListWidgetItem, QGraphicsRectItem,
    QTextEdit, QDialogButtonBox, QTableWidget, QTableWidgetItem, QSpinBox,
    QFormLayout, QDoubleSpinBox, QProgressDialog, QCheckBox, QStyle, QGroupBox, QGridLayout,
    QSizePolicy, QHeaderView, QAbstractScrollArea, QFrame
)
from PySide6.QtGui import (
    QPixmap, QImage, QPen, QColor, QAction, QCursor, QPainter, QFont, QMouseEvent,
    QKeySequence, QShortcut
)
from PySide6.QtCore import Qt, QRectF, Signal, QSize, QTimer, QPointF

from style import MATERIAL_STYLE

# ─── utils ─────────────────────────────────────────────────────────────────────
from utils import (
    # IO / paths
    safe_imread, list_images, load_annotations, save_annotations,
    extract_frames_from_video,

    # Navigation / service
    is_frame_annotated, find_next_matching_index, colorize_list_item,

    # Interpolation
    interpolate_range,

    # Geometry
    bbox_from_points,

    # YOLO helpers
    yolo_infer, yolo_best_bbox, annotate_points_and_bbox,

    # Export
    make_yolo_line, train_val_split, write_yolo_dataset_yaml,

    # Snapshots / QC
    snapshot_annotation, restore_snapshot, ensure_bbox_present
)


# ==============================================================================
# Dialog: load data
# ==============================================================================
class InitialLoadDialog(QDialog):
    """
    Dialog that allows choosing an image folder or a video file.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Data Source")
        layout = QVBoxLayout(self)

        self.result_path: Optional[str] = None

        self.folder_button = QPushButton("Select image folder")
        self.folder_button.clicked.connect(self.select_folder)
        layout.addWidget(self.folder_button)

        self.video_button = QPushButton("Select video file")
        self.video_button.clicked.connect(self.select_video)
        layout.addWidget(self.video_button)

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select an image folder")
        if not folder:
            return
        if not os.listdir(folder):
            QMessageBox.warning(self, "Warning", "No images found in the selected folder.")
            return
        self.result_path = folder
        self.accept()

    def select_video(self):
        video_path, _ = QFileDialog.getOpenFileName(
            self, "Select a video file", "",
            "Video files (*.mp4 *.avi *.mov)"
        )
        if not video_path:
            return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            QMessageBox.critical(self, "Error", f"Failed to open video:\n{video_path}")
            return
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if total_frames == 0:
            QMessageBox.critical(self, "Error", "Video file has no frames.")
            return

        # ── Extraction options
        opt_dialog = QDialog(self)
        opt_dialog.setWindowTitle("Frame extraction settings")
        dlayout = QVBoxLayout(opt_dialog)

        max_label = QLabel(f"Max frames: {total_frames}")
        dlayout.addWidget(max_label)

        form_layout = QFormLayout()
        mode_combo = QComboBox()
        mode_combo.addItems(["Step", "Sequential", "All"])
        form_layout.addRow("Extraction mode:", mode_combo)

        count_spin = QSpinBox()
        count_spin.setRange(1, total_frames)
        count_spin.setValue(min(10, total_frames))
        form_layout.addRow("Number of frames:", count_spin)
        dlayout.addLayout(form_layout)

        btn_layout = QHBoxLayout()
        extract_btn = QPushButton("Extract")
        cancel_btn = QPushButton("Cancel")
        btn_layout.addWidget(extract_btn)
        btn_layout.addWidget(cancel_btn)
        dlayout.addLayout(btn_layout)

        result = {"mode": "Step", "count": min(10, total_frames)}

        def do_extract():
            result["mode"] = mode_combo.currentText()
            result["count"] = count_spin.value()
            opt_dialog.accept()

        extract_btn.clicked.connect(do_extract)
        cancel_btn.clicked.connect(opt_dialog.reject)

        if opt_dialog.exec() != QDialog.Accepted:
            return

        # ── Run extraction utility
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        frames_dir = os.path.join(os.getcwd(), f"{base_name}_frames")
        os.makedirs(frames_dir, exist_ok=True)

        mode_map = {"Step": "step", "Sequential": "sequential", "All": "all"}
        mode = mode_map.get(result["mode"], "step")
        count = result["count"]

        paths = extract_frames_from_video(video_path, mode=mode, count=count, out_dir=frames_dir)
        if not paths:
            QMessageBox.warning(self, "Warning", "Failed to extract frames from video.")
            return

        self.result_path = frames_dir
        self.accept()


# ==============================================================================
# Dialog: skeleton configuration
# ==============================================================================
class SkeletonSetupDialog(QDialog):
    """
    Dialog for configuring skeleton keypoints and connections.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Skeleton setup")

        self.keypoints: List[str] = []
        self.connections: List[Tuple[str, str]] = []

        main_layout = QVBoxLayout(self)

        load_from_file_btn = QPushButton("Load skeleton from JSON")
        load_from_file_btn.clicked.connect(self.load_skeleton_from_file)

        self.points_edit = QTextEdit()
        self.points_edit.setPlaceholderText("Enter keypoint names, one per line")

        self.connections_edit = QTextEdit()
        self.connections_edit.setPlaceholderText(
            "Enter connections, e.g.:\nLeft_shoulder-Left_elbow"
        )

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept_skeleton)
        buttons.rejected.connect(self.reject)

        main_layout.addWidget(load_from_file_btn)
        main_layout.addWidget(QLabel("Keypoints:"))
        main_layout.addWidget(self.points_edit)
        main_layout.addWidget(QLabel("Connections:"))
        main_layout.addWidget(self.connections_edit)
        main_layout.addWidget(buttons)
        self.setLayout(main_layout)

    def load_skeleton_from_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select skeleton JSON", "", "JSON Files (*.json);;All Files (*)"
        )
        if not file_path:
            return

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to read file:\n{e}")
            return

        loaded_keypoints = data.get("keypoints", [])
        loaded_connections = data.get("connections", [])

        connections_lines = []
        for pair in loaded_connections:
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                a, b = pair
                connections_lines.append(f"{a}-{b}")

        self.points_edit.setPlainText("\n".join(loaded_keypoints))
        self.connections_edit.setPlainText("\n".join(connections_lines))

        QMessageBox.information(self, "Loaded", "Skeleton imported from file.")

    def accept_skeleton(self):
        points = [p.strip() for p in self.points_edit.toPlainText().split("\n") if p.strip()]
        connections_lines = [c.strip() for c in self.connections_edit.toPlainText().split("\n") if c.strip()]

        connections: List[Tuple[str, str]] = []
        for line in connections_lines:
            if "-" in line:
                a, b = [s.strip() for s in line.split("-", 1)]
                if a in points and b in points:
                    connections.append((a, b))
                else:
                    QMessageBox.warning(self, "Error", f"Connection has unknown points: {line}")
                    return
            else:
                QMessageBox.warning(self, "Error", f"Invalid connection format: {line}")
                return

        self.keypoints = points
        self.connections = connections
        self.accept()


# ==============================================================================
# Interactive bbox (drag + resize) with on_begin/on_end hooks
# ==============================================================================
class ResizableRectItem(QGraphicsRectItem):
    HANDLE_MARGIN = 6
    def __init__(self, x: float, y: float, w: float, h: float,
                 on_begin=None, on_change=None, on_end=None):
        super().__init__(0, 0, w, h)
        self.setPos(QPointF(x, y))
        self.on_begin = on_begin
        self.on_change = on_change
        self.on_end = on_end
        self.setFlags(
            QGraphicsRectItem.ItemIsMovable |
            QGraphicsRectItem.ItemIsSelectable |
            QGraphicsRectItem.ItemSendsGeometryChanges
        )
        self.setZValue(5)
        self.setPen(QPen(QColor(0, 200, 255), 2, Qt.DashLine))
        self.setCursor(QCursor(Qt.SizeAllCursor))
        self.setAcceptHoverEvents(True)

        self._resizing = False
        self._moving = False
        self._resize_edges = {"l": False, "r": False, "t": False, "b": False}
        self._last_scene_pos = None

    def _bbox_modifier_active(self) -> bool:
        return bool(QApplication.keyboardModifiers() & Qt.AltModifier)

    def _edge_under_cursor(self, scene_pos: QPointF):
        p = self.mapFromScene(scene_pos)
        r = self.rect()
        m = self.HANDLE_MARGIN
        edges = {"l": False, "r": False, "t": False, "b": False}
        if abs(p.x() - r.left())   <= m: edges["l"] = True
        if abs(p.x() - r.right())  <= m: edges["r"] = True
        if abs(p.y() - r.top())    <= m: edges["t"] = True
        if abs(p.y() - r.bottom()) <= m: edges["b"] = True
        return edges

    def hoverMoveEvent(self, event):
        if not self._bbox_modifier_active():
            self.setCursor(Qt.ArrowCursor)
            return
        edges = self._edge_under_cursor(event.scenePos())
        if (edges["l"] and edges["t"]) or (edges["r"] and edges["b"]):
            self.setCursor(Qt.SizeFDiagCursor)
        elif (edges["r"] and edges["t"]) or (edges["l"] and edges["b"]):
            self.setCursor(Qt.SizeBDiagCursor)
        elif edges["l"] or edges["r"]:
            self.setCursor(Qt.SizeHorCursor)
        elif edges["t"] or edges["b"]:
            self.setCursor(Qt.SizeVerCursor)
        else:
            self.setCursor(Qt.SizeAllCursor)
        super().hoverMoveEvent(event)

    def mousePressEvent(self, event):
        if not self._bbox_modifier_active():
            event.ignore()
            return
        edges = self._edge_under_cursor(event.scenePos())
        self._moving = not any(edges.values())
        if any(edges.values()):
            self._resizing = True
            self._resize_edges = edges
            self._last_scene_pos = event.scenePos()
            if callable(self.on_begin):
                self.on_begin()
            event.accept()
            return
        else:
            if callable(self.on_begin):
                self.on_begin()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if not self._bbox_modifier_active():
            event.ignore()
            return
        if self._resizing and self._last_scene_pos is not None:
            delta = event.scenePos() - self._last_scene_pos
            self._last_scene_pos = event.scenePos()
            x = self.pos().x(); y = self.pos().y()
            w = self.rect().width(); h = self.rect().height()
            if self._resize_edges["l"]: x += delta.x(); w -= delta.x()
            if self._resize_edges["r"]: w += delta.x()
            if self._resize_edges["t"]: y += delta.y(); h -= delta.y()
            if self._resize_edges["b"]: h += delta.y()
            w = max(1.0, w); h = max(1.0, h)
            self.setPos(QPointF(x, y)); self.setRect(0, 0, w, h)
            if callable(self.on_change):
                self.on_change(x, y, w, h)
            event.accept(); return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._resizing = False
        self._moving = False
        self._resize_edges = {"l": False, "r": False, "t": False, "b": False}
        self._last_scene_pos = None
        if callable(self.on_end):
            self.on_end()
        super().mouseReleaseEvent(event)

    def itemChange(self, change, value):
        if change == QGraphicsRectItem.ItemPositionHasChanged and not self._resizing:
            x = self.pos().x(); y = self.pos().y(); r = self.rect()
            if callable(self.on_change):
                self.on_change(x, y, r.width(), r.height())
        return super().itemChange(change, value)

    def set_rect_and_pos(self, x: float, y: float, w: float, h: float):
        self.setRect(0, 0, max(1.0, w), max(1.0, h))
        self.setPos(QPointF(x, y))


# ==============================================================================
# Image widget (points + bbox)
# ==============================================================================
class ImageGraphicsView(QGraphicsView):
    """
    Widget that renders the image, keypoints, and bbox overlays.
    """
    pointPlaced = Signal(float, float)
    pointMoved = Signal(str, float, float)
    bboxChanged = Signal(int, int, int, int)
    bboxEditStarted = Signal()
    bboxEditFinished = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.pixmap_item = None
        self.image = None

        self.keypoints: List[str] = []
        self.connections: List[Tuple[str, str]] = []

        self.points = {}  # {kp_name: (x,y)}
        self.selected_keypoint: Optional[str] = None
        self.hovered_point_name: Optional[str] = None
        self.dragging_point_name: Optional[str] = None

        # bbox state
        self.show_bbox = True
        self.bbox: Optional[List[int]] = None  # [x1, y1, x2, y2]
        self._bbox_item: Optional[ResizableRectItem] = None

        self._point_items = []
        self._conn_items = []

        self.setMouseTracking(True)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setCursor(QCursor(Qt.CrossCursor))

        # temporary drawing state
        self._bbox_drawing = False
        self._bbox_start = None

    def image_shape(self) -> Optional[Tuple[int, int]]:
        if self.image is None:
            return None
        return (self.image.shape[0], self.image.shape[1])

    def wheelEvent(self, event):
        scale_factor = 1.25
        if event.angleDelta().y() > 0:
            self.scale(scale_factor, scale_factor)
        else:
            self.scale(1 / scale_factor, 1 / scale_factor)

    def set_bbox(self, bbox: Optional[List[int]]):
        self.bbox = bbox
        self._ensure_bbox_item()
        self.update_view()

    def get_bbox(self) -> Optional[List[int]]:
        return self.bbox

    def _bbox_modifier_active(self, modifiers: Qt.KeyboardModifiers) -> bool:
        return bool(modifiers & Qt.AltModifier)

    def mousePressEvent(self, event: QMouseEvent):
        left_click = event.button() == Qt.LeftButton
        ctrl_pressed = bool(event.modifiers() & Qt.ControlModifier)
        bbox_mode = self._bbox_modifier_active(event.modifiers())

        # draw bbox only when Alt is held and there is no existing bbox
        if bbox_mode and left_click and not ctrl_pressed:
            if self.bbox is None:
                pos = self.mapToScene(event.position().toPoint())
                self._bbox_start = (pos.x(), pos.y())
                self._bbox_drawing = True
                self.bboxEditStarted.emit()
                return

        # pan with ctrl + left click
        if left_click and ctrl_pressed:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            super().mousePressEvent(event)
            return

        if left_click:
            item = self.itemAt(event.position().toPoint())
            if isinstance(item, ResizableRectItem):
                if bbox_mode:
                    super().mousePressEvent(event)
                    return
                # fall through to point handling when Alt is not held
                item = None

            pos = self.mapToScene(event.position().toPoint())
            x, y = pos.x(), pos.y()
            if self.keypoints and not bbox_mode:
                if self.hovered_point_name is not None:
                    self.dragging_point_name = self.hovered_point_name
                    self.setCursor(QCursor(Qt.ClosedHandCursor))
                else:
                    if (self.selected_keypoint is None
                            or self.points.get(self.selected_keypoint) is not None):
                        next_point = self._get_next_empty_point()
                        if next_point is None:
                            super().mousePressEvent(event)
                            return
                        self.selected_keypoint = next_point

                    if self.selected_keypoint is not None:
                        self.points[self.selected_keypoint] = (x, y)
                        self.update_view()
                        self.pointPlaced.emit(x, y)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        bbox_mode = self._bbox_modifier_active(event.modifiers())
        if self._bbox_drawing:
            if not bbox_mode:
                return
            pos = self.mapToScene(event.position().toPoint())
            x1, y1 = self._bbox_start
            x2, y2 = pos.x(), pos.y()
            # temporary rectangle preview
            if self._bbox_item is None:
                self._bbox_item = ResizableRectItem(
                    min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1),
                    on_begin=lambda: self.bboxEditStarted.emit(),
                    on_change=self._on_bbox_changed_by_item,
                    on_end=lambda: self.bboxEditFinished.emit()
                )
                self.scene().addItem(self._bbox_item)
            else:
                self._bbox_item.set_rect_and_pos(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
            return

        if self.dragMode() == QGraphicsView.ScrollHandDrag:
            super().mouseMoveEvent(event)
            return

        pos = self.mapToScene(event.position().toPoint())
        x, y = pos.x(), pos.y()
        if self.dragging_point_name is not None:
            self.points[self.dragging_point_name] = (x, y)
            self.update_view()
        else:
            if bbox_mode:
                self.setCursor(QCursor(Qt.CrossCursor))
                self.hovered_point_name = None
            elif self.keypoints:
                hovered = self._find_point_under_cursor(x, y)
                self.setCursor(QCursor(Qt.OpenHandCursor if hovered is not None else Qt.CrossCursor))
                self.hovered_point_name = hovered
            else:
                self.setCursor(QCursor(Qt.CrossCursor))
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self._bbox_drawing and event.button() == Qt.LeftButton:
            pos = self.mapToScene(event.position().toPoint())
            x1, y1 = self._bbox_start
            x2, y2 = pos.x(), pos.y()
            self._bbox_drawing = False
            self._bbox_start = None
            bx1, by1 = int(min(x1, x2)), int(min(y1, y2))
            bx2, by2 = int(max(x1, x2)), int(max(y1, y2))
            self.bbox = [bx1, by1, bx2, by2]
            if self._bbox_item is None:
                self._bbox_item = ResizableRectItem(
                    bx1, by1, bx2 - bx1, by2 - by1,
                    on_begin=lambda: self.bboxEditStarted.emit(),
                    on_change=self._on_bbox_changed_by_item,
                    on_end=lambda: self.bboxEditFinished.emit()
                )
                self.scene().addItem(self._bbox_item)
            else:
                self._bbox_item.set_rect_and_pos(bx1, by1, bx2 - bx1, by2 - by1)
            self.bboxChanged.emit(*self.bbox)
            self.update_view()
            self.bboxEditFinished.emit()  # finished drawing gesture
            return

        if self.dragMode() == QGraphicsView.ScrollHandDrag:
            self.setDragMode(QGraphicsView.NoDrag)
            super().mouseReleaseEvent(event)
            return

        if event.button() == Qt.LeftButton and self.dragging_point_name is not None:
            px, py = self.points[self.dragging_point_name]
            self.pointMoved.emit(self.dragging_point_name, px, py)
            self.dragging_point_name = None
            self.setCursor(QCursor(Qt.CrossCursor))
        super().mouseReleaseEvent(event)

    def _on_bbox_changed_by_item(self, x: float, y: float, w: float, h: float):
        bx1, by1 = int(x), int(y)
        bx2, by2 = int(x + w), int(y + h)
        self.bbox = [bx1, by1, bx2, by2]
        self.bboxChanged.emit(*self.bbox)

    def _ensure_bbox_item(self):
        if not self.show_bbox:
            return
        if self.bbox:
            x1, y1, x2, y2 = self.bbox
            x, y, w, h = min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)
            if self._bbox_item is None:
                self._bbox_item = ResizableRectItem(
                    x, y, w, h,
                    on_begin=lambda: self.bboxEditStarted.emit(),
                    on_change=self._on_bbox_changed_by_item,
                    on_end=lambda: self.bboxEditFinished.emit()
                )
                self.scene().addItem(self._bbox_item)
            else:
                # update geometry if the item already exists
                self._bbox_item.set_rect_and_pos(x, y, w, h)
            self._bbox_item.show()
        else:
            if self._bbox_item is not None:
                self.scene().removeItem(self._bbox_item)
                self._bbox_item = None

    def set_skeleton(self, keypoints: List[str], connections: List[Tuple[str, str]]):
        self.keypoints = keypoints
        self.connections = connections
        self.points = {kp: None for kp in self.keypoints}
        self.selected_keypoint = None
        self.update_view()

    def set_selected_keypoint(self, kp: str):
        self.selected_keypoint = kp
        self.update_view()

    def load_image(self, path: str):
        self.image = safe_imread(path)
        if self.image is None:
            QMessageBox.critical(self, "Error", f"Failed to load image: {path}")
            return
        self._show_image(self.image)

    def _show_image(self, image):
        # Scene fully resets — drop the bbox item reference.
        self.scene().clear()
        self._point_items.clear()
        self._conn_items.clear()
        self._bbox_item = None  # avoid dangling reference to deleted C++ object

        height, width, ch = image.shape
        bytesPerLine = ch * width
        qimg = QImage(image.data, width, height, bytesPerLine, QImage.Format_BGR888)
        pix = QPixmap.fromImage(qimg)
        self.pixmap_item = self.scene().addPixmap(pix)
        self.setSceneRect(QRectF(pix.rect()))
        self._draw_primitives()
        self._ensure_bbox_item()

    def _draw_primitives(self):
        if self.image is None:
            return
        # remove previous overlays (points/lines)
        for it in self._point_items:
            self.scene().removeItem(it)
        for it in self._conn_items:
            self.scene().removeItem(it)
        self._point_items.clear()
        self._conn_items.clear()

        # connections
        pen_conn = QPen(QColor(0, 255, 0), 2)
        hl_conn  = QPen(QColor(0, 200, 255), 3)

        for a, b in self.connections:
            if self.points.get(a) is not None and self.points.get(b) is not None:
                x1, y1 = self.points[a]
                x2, y2 = self.points[b]
                pen = hl_conn if self.selected_keypoint in (a, b) or self.hovered_point_name in (a, b) else pen_conn
                line = self.scene().addLine(x1, y1, x2, y2, pen)
                self._conn_items.append(line)

        # points
        base_pen = QPen(QColor(255, 0, 0), 3)
        sel_pen  = QPen(QColor(0, 200, 255), 3)
        hov_pen  = QPen(QColor(255, 200, 0), 3)

        for p_name, coords in self.points.items():
            if coords is None:
                continue
            x, y = coords
            radius = 6
            pen = base_pen
            fill = QColor(255, 0, 0, 160)
            if p_name == self.selected_keypoint:
                pen = sel_pen
                fill = QColor(0, 200, 255, 160)
                radius = 7
            elif p_name == self.hovered_point_name:
                pen = hov_pen
                fill = QColor(255, 200, 0, 160)
                radius = 7
            ellipse = self.scene().addEllipse(x - radius, y - radius, radius*2, radius*2, pen)
            ellipse.setBrush(fill)
            ellipse.setZValue(10)
            self._point_items.append(ellipse)
            if self.keypoints:
                text_item = self.scene().addText(p_name)
                text_item.setDefaultTextColor(QColor(255, 255, 255))
                text_item.setPos(x+8, y-8)
                text_item.setFont(QFont("Arial", 10))
                text_item.setZValue(10)
                self._point_items.append(text_item)

    def update_view(self):
        if self.image is not None:
            self._draw_primitives()
            self._ensure_bbox_item()

    def _get_next_empty_point(self):
        for kp in self.keypoints:
            if self.points.get(kp) is None:
                return kp
        return None

    def _find_point_under_cursor(self, x, y, threshold=10):
        for p_name, coords in self.points.items():
            if coords is not None:
                px, py = coords
                if (px - x)**2 + (py - y)**2 <= threshold**2:
                    return p_name
        return None

    def delete_point(self):
        if self.hovered_point_name is not None:
            self.points[self.hovered_point_name] = None
            self.hovered_point_name = None
            self.dragging_point_name = None
            self.update_view()


# ==============================================================================
# Main window
# ==============================================================================
class AnnotationMainWindow(QMainWindow):
    """
    Main annotation window.
    """
    def __init__(self, image_paths: List[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Pose annotator")
        self.resize(1280, 820)

        self.image_paths = image_paths
        self.current_index = 0

        # self.annotations[path] = {"points": {...}, "bbox": [x1,y1,x2,y2]}
        self.annotations = {}

        self.keypoints: List[str] = []
        self.connections: List[Tuple[str, str]] = []

        self.annotation_file_path: Optional[str] = None

        # YOLO model
        self.model = None
        self.device = 'cpu'
        self.auto_annotate_on_select = False

        # Image widget
        self.image_view = ImageGraphicsView()
        self.image_view.pointPlaced.connect(self.point_placed)
        self.image_view.pointMoved.connect(self.point_moved)
        self.image_view.bboxChanged.connect(self.on_bbox_changed)

        # Frames list (left dock)
        self.frames_list = QListWidget()
        for p in self.image_paths:
            item = QListWidgetItem(os.path.basename(p))
            self.frames_list.addItem(item)
        self.frames_list.currentRowChanged.connect(self.load_frame_by_index)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        # ── Model and auto-annotation controls
        controls_group = QGroupBox("Model & auto-annotation")
        cg = QGridLayout(controls_group)
        cg.setContentsMargins(8, 8, 8, 8)
        cg.setHorizontalSpacing(8)
        cg.setVerticalSpacing(6)

        self.btn_load_model = QPushButton("Load model")
        self.btn_load_model.setToolTip("Pick a .pt file and load YOLO")
        self.btn_load_model.setFixedHeight(30)
        self.btn_load_model.clicked.connect(self.on_load_model)

        self.btn_unload_model = QPushButton("Unload model")
        self.btn_unload_model.setToolTip("Free the model from memory")
        self.btn_unload_model.setFixedHeight(30)
        self.btn_unload_model.clicked.connect(self.on_unload_model)

        self.chk_auto_annot = QCheckBox("Auto on select")
        self.chk_auto_annot.setToolTip("If a frame has no annotations, run auto annotation when selected")
        self.chk_auto_annot.toggled.connect(self.toggle_auto_annotate_on_select)

        self.btn_auto_annot_all = QPushButton("Auto annotate all")
        self.btn_auto_annot_all.setToolTip("Run auto annotation for every frame")
        self.btn_auto_annot_all.setFixedHeight(30)
        self.btn_auto_annot_all.clicked.connect(self.auto_annotate_all)

        cg.addWidget(self.btn_load_model,   0, 0)
        cg.addWidget(self.btn_unload_model, 0, 1)
        cg.addWidget(self.chk_auto_annot,   1, 0)
        cg.addWidget(self.btn_auto_annot_all, 1, 1)

        # ── Inference controls (device/conf + mode label)
        infer_group = QGroupBox("Inference")
        ig = QGridLayout(infer_group)
        ig.setContentsMargins(8, 8, 8, 8)
        ig.setHorizontalSpacing(8)
        ig.setVerticalSpacing(6)

        dev_lbl = QLabel("Device:")
        self.device_combo = QComboBox()
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")
        self.device_combo.addItems(devices)
        self.device_combo.setToolTip("Select device (CPU/CUDA)")
        self.device_combo.setFixedWidth(100)

        conf_lbl = QLabel("Conf:")
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.0, 1.0)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.5)
        self.conf_spin.setToolTip("Confidence threshold")
        self.conf_spin.setFixedWidth(80)

        ig.addWidget(dev_lbl,           0, 0)
        ig.addWidget(self.device_combo, 0, 1)
        ig.addWidget(conf_lbl,          0, 2)
        ig.addWidget(self.conf_spin,    0, 3)
        ig.setColumnStretch(4, 1)

        # ── Frame list + navigation
        left_layout.addWidget(controls_group)
        left_layout.addWidget(infer_group)
        left_layout.addWidget(self.frames_list, 1)

        nav_group = QGroupBox("Navigation")
        ng = QGridLayout(nav_group)
        ng.setContentsMargins(8, 8, 8, 8)
        ng.setHorizontalSpacing(8)
        ng.setVerticalSpacing(6)

        prev_btn = QPushButton("Previous")
        next_btn = QPushButton("Next")
        interp_btn = QPushButton("Interpolate")
        clear_btn = QPushButton("Clear all")

        for b in (prev_btn, next_btn, interp_btn, clear_btn):
            b.setFixedHeight(28)
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        prev_btn.clicked.connect(self.prev_frame)
        next_btn.clicked.connect(self.next_frame)
        interp_btn.clicked.connect(self.interpolate_missing_all)
        clear_btn.clicked.connect(self.clear_all_annotations)

        ng.addWidget(prev_btn,   0, 0)
        ng.addWidget(next_btn,   0, 1)
        ng.addWidget(interp_btn, 1, 0)
        ng.addWidget(clear_btn,  1, 1)
        ng.setColumnStretch(0, 1)
        ng.setColumnStretch(1, 1)

        left_layout.addWidget(nav_group)

        dock_left = QDockWidget("Frames")
        dock_left.setWidget(left_widget)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock_left)

        # ── Right dock: compact tables
        right_container = QWidget()
        right_v = QVBoxLayout(right_container)
        right_v.setContentsMargins(6, 6, 6, 6)
        right_v.setSpacing(0)

        # BBox
        self.bbox_table = QTableWidget(1, 5)
        self.bbox_table.setHorizontalHeaderLabels(["BBox", "X1", "Y1", "X2", "Y2"])
        self.bbox_table.verticalHeader().setVisible(False)
        self.bbox_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.bbox_table.setShowGrid(False)
        self.bbox_table.setWordWrap(False)
        self.bbox_table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.bbox_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.bbox_table.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        hh = self.bbox_table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.Fixed)
        hh.setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.bbox_table.setColumnWidth(0, 60)
        for c in range(1, 5):
            self.bbox_table.setColumnWidth(c, 56)

        it0 = QTableWidgetItem("bbox")
        it0.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.bbox_table.setItem(0, 0, it0)
        for c in range(1, 5):
            it = QTableWidgetItem("")
            it.setFlags(it.flags() & ~Qt.ItemIsEditable)
            it.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            self.bbox_table.setItem(0, c, it)

        right_v.addWidget(self.bbox_table)
        clear_bbox_btn = QPushButton("Clear BBox")
        clear_bbox_btn.setFixedHeight(26)
        clear_bbox_btn.clicked.connect(self.clear_current_bbox)
        right_v.addWidget(clear_bbox_btn)
        self._fit_bbox_table_height()

        # Thin separator
        sep = QFrame()
        sep.setObjectName("bbox_points_separator")
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Plain)
        sep.setLineWidth(1)
        sep.setMidLineWidth(0)
        sep.setStyleSheet("""
        #bbox_points_separator {
            background-color: #3A3A3A;
            min-height: 1px; max-height: 1px;
            margin-top: 4px; margin-bottom: 4px;
            border: none;
        }
        """)
        right_v.addWidget(sep)

        # Points table
        self.points_table = QTableWidget()
        self.points_table.setColumnCount(3)
        self.points_table.setHorizontalHeaderLabels(["Keypoint", "X", "Y"])
        self.points_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.points_table.setShowGrid(False)
        self.points_table.setWordWrap(False)
        self.points_table.verticalHeader().setVisible(False)

        hp = self.points_table.horizontalHeader()
        hp.setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        hp.setSectionResizeMode(0, QHeaderView.Stretch)
        hp.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        hp.setSectionResizeMode(2, QHeaderView.ResizeToContents)

        # Enable selecting a keypoint by clicking a row
        self.points_table.cellClicked.connect(self.select_keypoint_from_table)

        right_v.addWidget(self.points_table, 1)

        dock_right = QDockWidget("Details")
        dock_right.setWidget(right_container)
        dock_right.setMinimumWidth(280)
        self.addDockWidget(Qt.RightDockWidgetArea, dock_right)

        # ── Central area — image only
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.image_view)
        self.setCentralWidget(main_widget)

        # ── Menu
        menu_bar = QMenuBar()
        menu_bar.setStyleSheet(MATERIAL_STYLE)
        file_menu = menu_bar.addMenu("File")

        setup_skel_act = QAction("Configure skeleton", self)
        setup_skel_act.triggered.connect(self.setup_skeleton)
        file_menu.addAction(setup_skel_act)

        save_ann_act = QAction("Save annotations as...", self)
        save_ann_act.triggered.connect(self.save_annotations_as)
        file_menu.addAction(save_ann_act)

        load_ann_act = QAction("Load annotations", self)
        load_ann_act.triggered.connect(self.load_annotations_dialog)
        file_menu.addAction(load_ann_act)

        file_menu.addSeparator()
        act_yolo = QAction("Save as YOLO", self); act_yolo.triggered.connect(self.save_yolo_format)
        file_menu.addAction(act_yolo)

        self.setMenuBar(menu_bar)

        # ── Shortcuts
        QShortcut(QKeySequence("Ctrl+S"), self, activated=self.quick_save)
        # Undo / Redo
        QShortcut(QKeySequence("Ctrl+Z"), self, activated=self.undo)
        QShortcut(QKeySequence("Ctrl+Y"), self, activated=self.redo)

        # ── Undo/Redo stacks
        self.undo_stack: List[Tuple[str, dict]] = []
        self.redo_stack: List[Tuple[str, dict]] = []

        # ── Autosave
        self.autosave_enabled = True
        self.autosave_timer = QTimer(self)
        self.autosave_timer.setInterval(60_000)
        self.autosave_timer.timeout.connect(self.autosave_tick)
        self.autosave_timer.start()
        self.autosave_path = os.path.join(os.path.dirname(self.image_paths[0]) if self.image_paths else os.getcwd(),
                                          "autosave_annotations.json")

        # Single undo step per bbox edit gesture
        self._bbox_edit_active = False
        self.image_view.bboxEditStarted.connect(self._bbox_edit_begin)
        self.image_view.bboxEditFinished.connect(self._bbox_edit_end)

        if self.frames_list.count() > 0:
            self.frames_list.setCurrentRow(0)
        self.update_frames_list_markers()

    # ──────────────────────────── YOLO ────────────────────────────
    def on_load_model(self):
        model_path, _ = QFileDialog.getOpenFileName(self, "Select YOLO model file", "", "Models (*.pt)")
        if model_path:
            device = self.device_combo.currentText()
            try:
                from ultralytics import YOLO
                self.model = YOLO(model_path)
                self.model.to(device)
                self.device = device
                QMessageBox.information(self, "Model", f"Loaded: {os.path.basename(model_path)}\nDevice: {device}")
            except Exception as e:
                self.model = None
                QMessageBox.critical(self, "Error", f"Failed to load model:\n{e}")

    def _fit_bbox_table_height(self):
        h_header = self.bbox_table.horizontalHeader().height()
        h_row    = self.bbox_table.sizeHintForRow(0)
        h_frame  = self.bbox_table.frameWidth() * 2
        self.bbox_table.setFixedHeight(h_header + h_row + h_frame + 2)

    def on_unload_model(self):
        self.model = None
        QMessageBox.information(self, "Model", "Model unloaded.")

    def is_model_loaded(self):
        return self.model is not None

    def get_conf_threshold(self) -> float:
        return self.conf_spin.value()

    # ───────────────────── Annotations / Undo / Redo ─────────────────────
    def _push_undo(self):
        path, snap = snapshot_annotation(self.annotations, self.image_paths[self.current_index])
        self.undo_stack.append((path, snap))
        self.redo_stack.clear()

    def _bbox_edit_begin(self):
        if not self._bbox_edit_active:
            self._push_undo()
            self._bbox_edit_active = True

    def _bbox_edit_end(self):
        self._bbox_edit_active = False

    def undo(self):
        if not self.undo_stack:
            return
        cur_path, cur_snap = snapshot_annotation(self.annotations, self.image_paths[self.current_index])
        path, snap = self.undo_stack.pop()
        self.redo_stack.append((cur_path, cur_snap))
        restore_snapshot(self.annotations, path, snap)
        if path == self.image_paths[self.current_index]:
            self.load_frame(self.current_index)
        self.update_frames_list_markers()

    def redo(self):
        if not self.redo_stack:
            return
        cur_path, cur_snap = snapshot_annotation(self.annotations, self.image_paths[self.current_index])
        path, snap = self.redo_stack.pop()
        self.undo_stack.append((cur_path, cur_snap))
        restore_snapshot(self.annotations, path, snap)
        if path == self.image_paths[self.current_index]:
            self.load_frame(self.current_index)
        self.update_frames_list_markers()

    def set_annotation(self, image_path: str, point_name: str, coords: Tuple[int, int]):
        if image_path not in self.annotations:
            self.annotations[image_path] = {"points": {}}
        self.annotations[image_path]["points"][point_name] = coords

    def get_annotation(self, image_path: str):
        return self.annotations.get(image_path, {}).get("points", {})

    def to_dict(self):
        return self.annotations

    # ─────────────────────────── Auto annotation ───────────────────────────
    def toggle_auto_annotate_on_select(self, state):
        self.auto_annotate_on_select = bool(state)

    def auto_annotate_all(self):
        if not self.is_model_loaded():
            QMessageBox.warning(self, "Warning", "Model is not loaded!")
            return
        if not self.keypoints:
            QMessageBox.warning(self, "Error", "Cannot auto annotate without a skeleton definition.")
            return

        progress = QProgressDialog("Auto annotation...", "Cancel", 0, len(self.image_paths), self)
        progress.setWindowTitle("Auto annotation")
        progress.setWindowModality(Qt.WindowModal)
        progress.setValue(0)

        canceled = False
        conf_val = self.get_conf_threshold()

        for i, path in enumerate(self.image_paths):
            progress.setValue(i)
            progress.setLabelText(f"Annotating frame {i+1} of {len(self.image_paths)}")
            if progress.wasCanceled():
                canceled = True
                break

            img = safe_imread(path)
            if img is None:
                continue

            combo = annotate_points_and_bbox(self.model, img, self.keypoints, conf=conf_val)
            if combo["points"]:
                for p_name, coords in combo["points"].items():
                    self.set_annotation(path, p_name, (int(coords[0]), int(coords[1])))
            if combo["bbox"] is not None:
                if path not in self.annotations:
                    self.annotations[path] = {"points": {}}
                self.annotations[path]["bbox"] = combo["bbox"]

        progress.setValue(len(self.image_paths))

        if canceled:
            QMessageBox.information(self, "Cancelled", "Auto annotation was cancelled by the user.")
        else:
            QMessageBox.information(self, "Done", "Auto annotation finished.")

        self.update_frames_list_markers()
        self.load_frame(self.current_index)

    # ───────────────────────── GUI handlers ─────────────────────────
    def point_placed(self, x, y):
        if self.keypoints and self.image_view.selected_keypoint is not None:
            p_name = self.image_view.selected_keypoint
            path = self.image_paths[self.current_index]
            self._push_undo()
            self.set_annotation(path, p_name, (int(x), int(y)))
            self.update_points_table()
            self.update_frames_list_markers()

    def point_moved(self, point_name, x, y):
        if point_name and self.keypoints:
            path = self.image_paths[self.current_index]
            self._push_undo()
            self.set_annotation(path, point_name, (int(x), int(y)))
            self.update_points_table()
            self.update_frames_list_markers()

    def on_bbox_changed(self, x1, y1, x2, y2):
        path = self.image_paths[self.current_index]
        if path not in self.annotations:
            self.annotations[path] = {"points": {}}
        self.annotations[path]["bbox"] = [int(x1), int(y1), int(x2), int(y2)]
        self.update_frames_list_markers()
        self.update_bbox_table([x1, y1, x2, y2])

    def load_frame_by_index(self, idx):
        if idx < 0 or idx >= len(self.image_paths):
            return
        self.load_frame(idx)

    def load_frame(self, idx):
        self.current_index = idx
        if not self.image_paths:
            return
        path = self.image_paths[idx]

        # Auto annotate on selection (if missing)
        if self.auto_annotate_on_select and self.is_model_loaded() and self.keypoints:
            ann_points = self.get_annotation(path)
            has_bbox = bool(self.annotations.get(path, {}).get("bbox"))
            if (not ann_points or not has_bbox):
                img = safe_imread(path)
                if img is not None:
                    conf_val = self.get_conf_threshold()
                    combo = annotate_points_and_bbox(self.model, img, self.keypoints, conf=conf_val)
                    if combo["points"]:
                        for p_name, coords in combo["points"].items():
                            self.set_annotation(path, p_name, (int(coords[0]), int(coords[1])))
                    if combo["bbox"] is not None:
                        if path not in self.annotations:
                            self.annotations[path] = {"points": {}}
                        self.annotations[path]["bbox"] = combo["bbox"]

        # Load image
        self.image_view.load_image(path)

        # Points
        ann = self.get_annotation(path)
        if self.keypoints:
            for kp in self.keypoints:
                coords = ann.get(kp, None)
                self.image_view.points[kp] = coords

        # BBox
        bbox = self.annotations.get(path, {}).get("bbox", None)
        self.image_view.set_bbox(bbox)

        self.image_view.update_view()
        self.update_points_table()
        self.update_frames_list_markers()
        self.update_bbox_table(bbox)

    def update_bbox_table(self, bbox: Optional[List[int]]):
        vals = ["", "", "", ""]
        if bbox:
            vals = [str(int(bbox[0])), str(int(bbox[1])),
                    str(int(bbox[2])), str(int(bbox[3]))]
        for i, v in enumerate(vals, start=1):
            it = self.bbox_table.item(0, i)
            if it is None:
                it = QTableWidgetItem(v)
                it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                self.bbox_table.setItem(0, i, it)
            else:
                it.setText(v)
            it.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._fit_bbox_table_height()

    def clear_current_bbox(self):
        if not self.image_paths:
            return
        path = self.image_paths[self.current_index]
        record = self.annotations.get(path)
        bbox = record.get("bbox") if isinstance(record, dict) else None
        if not bbox:
            return
        if not isinstance(record, dict):
            return
        self._push_undo()
        record["bbox"] = None
        self.image_view.set_bbox(None)
        self.update_bbox_table(None)

    def update_points_table(self):
        self.points_table.setRowCount(0)
        if self.keypoints:
            for kp in self.keypoints:
                row = self.points_table.rowCount()
                self.points_table.insertRow(row)
                name_item = QTableWidgetItem(kp)
                name_item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                self.points_table.setItem(row, 0, name_item)

                coords = self.image_view.points.get(kp)
                if coords is not None:
                    x, y = int(coords[0]), int(coords[1])
                    self.points_table.setItem(row, 1, QTableWidgetItem(str(x)))
                    self.points_table.setItem(row, 2, QTableWidgetItem(str(y)))
                else:
                    self.points_table.setItem(row, 1, QTableWidgetItem(""))
                    self.points_table.setItem(row, 2, QTableWidgetItem(""))

    def prev_frame(self):
        if self.current_index > 0:
            self.frames_list.setCurrentRow(self.current_index - 1)

    def next_frame(self):
        if self.current_index < len(self.image_paths) - 1:
            self.frames_list.setCurrentRow(self.current_index + 1)

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_W and (event.modifiers() & Qt.AltModifier):
            self.clear_current_bbox()
            return
        elif key == Qt.Key_A:
            self.prev_frame()
        elif key == Qt.Key_D:
            self.next_frame()
        elif key == Qt.Key_W:
            self._push_undo()
            self.image_view.delete_point()
            current_path = self.image_paths[self.current_index]
            points = self.get_annotation(current_path)
            for kp in self.keypoints:
                if self.image_view.points.get(kp) is None and kp in points:
                    points.pop(kp, None)
            self.update_points_table()
            self.update_frames_list_markers()
        else:
            super().keyPressEvent(event)

    # ───────────────────── Annotation navigation ─────────────────────
    def update_frames_list_markers(self):
        for i, p in enumerate(self.image_paths):
            item = self.frames_list.item(i)
            colorize_list_item(item, is_frame_annotated(self.annotations, p))

    def next_unannotated(self):
        idx = find_next_matching_index(
            self.image_paths, self.current_index,
            lambda p: not is_frame_annotated(self.annotations, p)
        )
        if idx is not None:
            self.frames_list.setCurrentRow(idx)

    def next_annotated(self):
        idx = find_next_matching_index(
            self.image_paths, self.current_index,
            lambda p: is_frame_annotated(self.annotations, p)
        )
        if idx is not None:
            self.frames_list.setCurrentRow(idx)

    # ───────────────────── Interpolation ─────────────────────
    def clear_all_annotations(self):
        reply = QMessageBox.question(
            self, "Confirmation",
            "Delete all current annotations?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self._push_undo()
            self.annotations.clear()
            if self.keypoints:
                for kp in self.keypoints:
                    self.image_view.points[kp] = None
            self.image_view.set_bbox(None)
            self.image_view.update_view()
            self.update_points_table()
            self.update_frames_list_markers()
            self.update_bbox_table(None)
            QMessageBox.information(self, "Done", "All annotations have been removed.")

    def interpolate_missing_all(self):
        if not self.image_paths or len(self.image_paths) < 2:
            QMessageBox.warning(self, "Error", "Not enough frames for interpolation.")
            return

        annotated_indices = []
        for i, path in enumerate(self.image_paths):
            points_dict = self.get_annotation(path)
            if len(points_dict) > 0:
                annotated_indices.append(i)

        if len(annotated_indices) < 2:
            QMessageBox.warning(self, "Error", "Need at least two annotated frames for interpolation.")
            return

        for idx in range(len(annotated_indices) - 1):
            start_idx = annotated_indices[idx]
            end_idx = annotated_indices[idx + 1]
            if end_idx - start_idx < 2:
                continue
            interpolate_range(self.image_paths, self.annotations, self.keypoints, start_idx, end_idx)

        QMessageBox.information(self, "Done", "All missing frames were interpolated.")
        self.load_frame(self.current_index)

    # ───────────────────── Save / load ─────────────────────
    def save_annotations_as(self):
        save_path, _ = QFileDialog.getSaveFileName(self, "Save annotations", "", "JSON Files (*.json)")
        if save_path:
            self.annotation_file_path = save_path
            self._save_annotations()

    def quick_save(self):
        if not self.annotation_file_path:
            self.save_annotations_as()
        else:
            self._save_annotations()

    def _save_annotations_dict(self) -> dict:
        for path, rec in self.annotations.items():
            ensure_bbox_present(rec)
        return {
            "keypoints": self.keypoints,
            "connections": self.connections,
            "annotations": self.annotations
        }

    def _save_annotations(self):
        if not self.annotation_file_path:
            return
        ok = save_annotations(self.annotation_file_path, self._save_annotations_dict())
        if ok:
            QMessageBox.information(self, "Saved", f"Annotations stored at:\n{self.annotation_file_path}")
        else:
            QMessageBox.critical(self, "Error", f"Failed to save annotations:\n{self.annotation_file_path}")

    def autosave_tick(self):
        if not self.autosave_enabled:
            return
        _ = save_annotations(self.autosave_path, self._save_annotations_dict())

    def load_annotations_dialog(self):
        load_path, _ = QFileDialog.getOpenFileName(self, "Load annotations", "", "JSON Files (*.json)")
        if not load_path:
            return
        self.annotation_file_path = load_path

        data = load_annotations(load_path)
        if not data:
            QMessageBox.warning(self, "Warning", "Annotation file is empty or corrupted.")
            return

        self.keypoints = data.get("keypoints", [])
        self.connections = data.get("connections", [])
        raw_annotations = data.get("annotations", {}) or {}

        if raw_annotations:
            remapped: Dict[str, Dict] = {}
            by_name: Dict[str, Dict] = {}
            for stored_path, rec in raw_annotations.items():
                by_name[os.path.basename(stored_path)] = rec
            for path in self.image_paths:
                name = os.path.basename(path)
                if name in by_name:
                    remapped[path] = by_name[name]
            self.annotations = remapped if remapped else raw_annotations
        else:
            self.annotations = {}

        self.image_view.set_skeleton(self.keypoints, self.connections)
        if not self.keypoints:
            resp = QMessageBox.question(
                self,
                "Skeleton",
                "No skeleton found in annotations. Configure it now?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if resp == QMessageBox.Yes:
                self.setup_skeleton()

        QMessageBox.information(self, "Loaded", "Annotations loaded.")
        self.load_frame(self.current_index)
        self.update_frames_list_markers()

    # ───────────────────── YOLO export ─────────────────────
    def save_yolo_format(self):
        if not self.keypoints:
            QMessageBox.warning(self, "Error", "Skeleton is not configured, nothing to export to YOLO.")
            return

        out_dir = QFileDialog.getExistingDirectory(self, "Select output folder for YOLO annotations")
        if not out_dir:
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Train/val split")
        vlayout = QVBoxLayout(dlg)
        form = QFormLayout()
        train_ratio_spin = QDoubleSpinBox()
        train_ratio_spin.setRange(0.0, 1.0)
        train_ratio_spin.setSingleStep(0.1)
        train_ratio_spin.setValue(0.8)
        form.addRow("Train fraction (0..1):", train_ratio_spin)
        vlayout.addLayout(form)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        vlayout.addWidget(bb)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        if dlg.exec() != QDialog.Accepted:
            return

        train_ratio = train_ratio_spin.value()

        images_dir = os.path.join(out_dir, "images")
        labels_dir = os.path.join(out_dir, "labels")
        images_train = os.path.join(images_dir, "train")
        images_val = os.path.join(images_dir, "val")
        labels_train = os.path.join(labels_dir, "train")
        labels_val = os.path.join(labels_dir, "val")
        os.makedirs(images_train, exist_ok=True)
        os.makedirs(images_val, exist_ok=True)
        os.makedirs(labels_train, exist_ok=True)
        os.makedirs(labels_val, exist_ok=True)

        train_set, val_set = train_val_split(self.image_paths, ratio=train_ratio)
        conf_val = self.get_conf_threshold()

        def prepare_line(path: str) -> Optional[str]:
            img = safe_imread(path)
            if img is None:
                return None
            H, W = img.shape[:2]
            saved_bbox = self.annotations.get(path, {}).get("bbox")
            bbox_xyxy = saved_bbox
            if bbox_xyxy is None and self.is_model_loaded():
                res = yolo_infer(self.model, img, conf_val)
                bbox_xyxy = yolo_best_bbox(res, (H, W))
            if bbox_xyxy is None:
                return None
            points = self.annotations.get(path, {}).get("points", {}) or {}
            line = make_yolo_line((H, W), bbox_xyxy, self.keypoints, points, cls_id=0)
            return line

        train_set = set(train_set)
        for path in self.image_paths:
            if not self.annotations.get(path, {}).get("points"):
                continue
            line = prepare_line(path)
            if not line:
                continue
            filename = os.path.basename(path)
            in_train = path in train_set
            dst_img = os.path.join(images_train if in_train else images_val, filename)
            dst_txt = os.path.join(labels_train if in_train else labels_val, filename.rsplit('.', 1)[0] + ".txt")
            shutil.copyfile(path, dst_img)
            with open(dst_txt, 'w', encoding='utf-8') as f:
                f.write(line + "\n")

        yaml_path = None
        try:
            yaml_path = write_yolo_dataset_yaml(out_dir, self.keypoints, self.connections)
        except Exception as e:
            QMessageBox.warning(
                self,
                "Warning",
                f"Annotations saved but dataset.yaml creation failed:\n{e}"
            )

        msg = "Exported to YOLO format (train/val split)."
        if yaml_path:
            msg += f"\nYAML: {yaml_path}"
        QMessageBox.information(self, "Done", msg)

    # ───────────────────── Skeleton ─────────────────────
    def setup_skeleton(self):
        dialog = SkeletonSetupDialog(self)
        if dialog.exec():
            self.keypoints = dialog.keypoints
            self.connections = dialog.connections
            self.image_view.set_skeleton(self.keypoints, self.connections)
            self.update_points_table()

    def select_keypoint_from_table(self, row, col):
        if self.keypoints:
            p_name_item = self.points_table.item(row, 0)
            if p_name_item is not None:
                p_name = p_name_item.text()
                self.image_view.set_selected_keypoint(p_name)

    def closeEvent(self, event):
        QApplication.instance().quit()
        event.accept()


# ==============================================================================
# main
# ==============================================================================
def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(MATERIAL_STYLE)

    dlg = InitialLoadDialog()
    if dlg.exec() == QDialog.Accepted:
        frames_dir = dlg.result_path
        if not frames_dir:
            sys.exit(0)

        image_files = list_images(frames_dir)
        if not image_files:
            QMessageBox.warning(None, "Warning", "No images in the selected folder.")
            sys.exit(0)

        w = AnnotationMainWindow(image_files)
        w.show()
        w.setFocus()
        sys.exit(app.exec())
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
