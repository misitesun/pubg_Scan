"""只操作本程序窗口与系统热键，无游戏进程句柄、注入或输入模拟。"""
import ctypes
from ctypes import wintypes
import sys
import time

from PySide6.QtCore import Qt, QRectF, QAbstractNativeEventFilter
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget


def user32():
    dll = ctypes.WinDLL("user32", use_last_error=True)
    dll.SetWindowDisplayAffinity.argtypes = [wintypes.HWND, wintypes.DWORD]
    dll.SetWindowDisplayAffinity.restype = wintypes.BOOL
    dll.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int,
                                ctypes.c_int, ctypes.c_int, wintypes.UINT]
    dll.SetWindowPos.restype = wintypes.BOOL
    dll.RegisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int, wintypes.UINT, wintypes.UINT]
    dll.RegisterHotKey.restype = wintypes.BOOL
    dll.UnregisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int]
    dll.UnregisterHotKey.restype = wintypes.BOOL
    dll.GetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int]
    dll.GetWindowLongPtrW.restype = ctypes.c_ssize_t
    dll.SetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_ssize_t]
    dll.SetWindowLongPtrW.restype = ctypes.c_ssize_t
    return dll


class Overlay(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint |
                            Qt.WindowType.Tool | Qt.WindowType.WindowTransparentForInput |
                            Qt.WindowType.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.boxes = []
        self.captured = 0
        self.region = None

    def prepare(self):
        if sys.platform != "win32":
            raise ValueError("透明屏幕画框仅支持 Windows。")
        self.resize(1, 1)
        self.show()
        dll, handle = user32(), int(self.winId())
        old = dll.GetWindowLongPtrW(handle, -20)
        dll.SetWindowLongPtrW(handle, -20, old | 0x80000 | 0x20 | 0x80 | 0x08000000)
        # 仅排除本程序画框窗口，避免把自己绘制的框再次输入模型。
        # 这不是隐藏进程或绕过反作弊，失败时不启动采集。
        if not dll.SetWindowDisplayAffinity(handle, 0x11):
            self.hide()
            raise ValueError("系统无法从采集中排除本程序画框窗口，屏幕测试未启动。")

    def set_boxes(self, region, boxes, captured):
        if region != self.region:
            if not user32().SetWindowPos(int(self.winId()), -1, region.left, region.top,
                                        region.width, region.height, 0x0010 | 0x0040):
                raise ValueError("覆盖层位置设置失败。")
            self.region = region
        self.boxes, self.captured = boxes, captured
        self.update()

    def clear(self):
        self.boxes = []
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
        painter.fillRect(self.rect(), Qt.GlobalColor.transparent)
        # 旧结果不长期残留；尚未实现运动预测。
        if time.perf_counter() - self.captured > 0.25:
            return
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        painter.setPen(QPen(QColor("#00ff99"), 2))
        ratio = self.devicePixelRatioF()
        for box in self.boxes:
            painter.drawRect(QRectF(box.x1/ratio, box.y1/ratio,
                                   (box.x2-box.x1)/ratio, (box.y2-box.y1)/ratio))


class StopHotkey(QAbstractNativeEventFilter):
    def __init__(self, callback):
        super().__init__()
        self.callback = callback
        self.registered = False

    def register(self):
        self.registered = bool(user32().RegisterHotKey(None, 0x5031, 0x4000, 0x77))  # F8
        if not self.registered:
            raise ValueError("F8 停止热键已被占用，请先释放该热键后再测试。")

    def close(self):
        if self.registered:
            user32().UnregisterHotKey(None, 0x5031)
            self.registered = False

    def nativeEventFilter(self, event_type, message):
        if sys.platform == "win32":
            message = wintypes.MSG.from_address(int(message))
            if message.message == 0x0312 and message.wParam == 0x5031:
                self.callback()
                return True, 0
        return False, 0
