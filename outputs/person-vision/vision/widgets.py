import cv2
from PySide6.QtCore import Qt, QPointF, QRectF, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QWidget


def qimage(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    return QImage(rgb.data, width, height, rgb.strides[0], QImage.Format.Format_RGB888).copy()


class ImageCanvas(QWidget):
    changed = Signal()

    def __init__(self, editable=False):
        super().__init__()
        self.editable = editable
        self.image = None
        self.boxes = []
        self.roles = []
        self.ignored_box_indices=set()
        self.new_box_role = 'unknown'
        self.drag_start = self.drag_end = None
        self.setMinimumSize(400, 280)
        self.setMouseTracking(True)

    def set_frame(self, frame, boxes=(), roles=None, ignored_indices=()):
        self.image = qimage(frame) if frame is not None else None
        self.boxes = [list(box) for box in boxes]
        self.roles = list(roles) if roles is not None and len(roles)==len(self.boxes) else ['unknown']*len(self.boxes)
        self.ignored_box_indices={i for i in ignored_indices if 0<=i<len(self.boxes)}
        self.drag_start = self.drag_end = None
        self.update()

    def image_rect(self):
        if self.image is None:
            return QRectF()
        ratio = min(self.width() / self.image.width(), self.height() / self.image.height())
        width, height = self.image.width() * ratio, self.image.height() * ratio
        return QRectF((self.width() - width) / 2, (self.height() - height) / 2, width, height)

    def image_point(self, point):
        rect = self.image_rect()
        if rect.isEmpty():
            return None
        return QPointF(max(0, min(self.image.width(), (point.x() - rect.left()) * self.image.width() / rect.width())),
                       max(0, min(self.image.height(), (point.y() - rect.top()) * self.image.height() / rect.height())))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#0e1522"))
        if self.image is None:
            painter.setPen(QColor("#8d9bb0"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "导入素材后在这里标注人物" if self.editable else "选择模型和录像，开始验证")
            return
        rect = self.image_rect()
        painter.drawImage(rect, self.image)
        painter.setPen(QPen(QColor("#52e4b5"), 2))
        sx, sy = rect.width() / self.image.width(), rect.height() / self.image.height()
        from .identity import ROLE_LABELS,ROLE_COLORS
        for index,(x1, y1, x2, y2) in enumerate(self.boxes):
            role=self.roles[index]
            ignored=index in self.ignored_box_indices
            painter.setPen(QPen(QColor('#9099a8' if ignored else ROLE_COLORS[role]),2))
            painter.drawRect(QRectF(rect.x() + x1*sx, rect.y() + y1*sy, (x2-x1)*sx, (y2-y1)*sy))
            painter.drawText(QPointF(rect.x()+x1*sx, max(rect.y()+15,rect.y()+y1*sy-5)),f'{index+1} '+('忽略（已死亡）' if ignored else ROLE_LABELS[role]))
        if self.drag_start is not None and self.drag_end is not None:
            painter.setPen(QPen(QColor("#ffc56b"), 2, Qt.PenStyle.DashLine))
            painter.drawRect(QRectF(self.drag_start, self.drag_end).normalized())

    def mousePressEvent(self, event):
        if self.editable and self.image and event.button()==Qt.MouseButton.RightButton and self.image_rect().contains(event.position()):
            point=self.image_point(event.position())
            # 重叠时优先选择面积更小的框，便于修改遮挡人物。
            candidates=[i for i,b in enumerate(self.boxes) if b[0]<=point.x()<=b[2] and b[1]<=point.y()<=b[3]]
            if candidates:
                i=min(candidates,key=lambda i:(self.boxes[i][2]-self.boxes[i][0])*(self.boxes[i][3]-self.boxes[i][1]))
                self.roles[i]=self.new_box_role;self.changed.emit();self.update()
            return
        if self.editable and self.image and event.button() == Qt.MouseButton.LeftButton:
            if self.image_rect().contains(event.position()):
                self.drag_start = self.drag_end = event.position()

    def mouseMoveEvent(self, event):
        if self.drag_start is not None:
            self.drag_end = event.position()
            self.update()

    def mouseReleaseEvent(self, event):
        if self.drag_start is None:
            return
        a, b = self.image_point(self.drag_start), self.image_point(event.position())
        x1, x2 = sorted((a.x(), b.x()))
        y1, y2 = sorted((a.y(), b.y()))
        if x2-x1 >= 2 and y2-y1 >= 2:
            self.boxes.append([x1, y1, x2, y2])
            self.roles.append(self.new_box_role)
            self.changed.emit()
        self.drag_start = self.drag_end = None
        self.update()

    def undo(self):
        if self.boxes:
            self.boxes.pop()
            self.roles.pop()
            self.ignored_box_indices.discard(len(self.boxes))
            self.changed.emit()
            self.update()

    def clear(self):
        self.boxes = []
        self.roles = []
        self.ignored_box_indices.clear()
        self.changed.emit()
        self.update()
