"""输入适配器；屏幕模式不打开游戏句柄、不记录画面。"""
from dataclasses import dataclass
import sys
import time

import cv2
import numpy as np


@dataclass(frozen=True)
class Region:
    left: int
    top: int
    width: int
    height: int


class VideoSource:
    def __init__(self, path):
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            self.capture.release()
            raise ValueError("录像无法打开，建议使用 MP4（H.264）。")
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(self.fps) or self.fps <= 0:
            self.fps = 30
        self.duration = self.capture.get(cv2.CAP_PROP_FRAME_COUNT) / self.fps
        self.started = None

    def read(self):
        # 按原录像时间取帧；推理跟不上时跳过旧帧，不堆积延迟。
        now = time.perf_counter()
        if self.started is None:
            self.started = now
        target = int((now - self.started) * self.fps)
        current = int(self.capture.get(cv2.CAP_PROP_POS_FRAMES))
        if target > current + 1:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, frame = self.capture.read()
        if not ok:
            return None
        height, width = frame.shape[:2]
        return frame, Region(0, 0, width, height), now

    def close(self):
        self.capture.release()


class ScreenSource:
    def __init__(self, monitor=1):
        if sys.platform != "win32":
            raise ValueError("屏幕画框原型目前只开放 Windows；Mac 可进行标注与录像验证。")
        import mss
        self.capture = mss.mss()
        if not 1 <= monitor < len(self.capture.monitors):
            self.capture.close()
            raise ValueError("显示器编号不存在（编号从 1 开始）。")
        self.monitor = self.capture.monitors[monitor]

    def read(self):
        pixels = np.asarray(self.capture.grab(self.monitor))
        frame = np.ascontiguousarray(pixels[:, :, :3])
        region = Region(**{key: self.monitor[key] for key in ("left", "top", "width", "height")})
        return frame, region, time.perf_counter()

    def close(self):
        self.capture.close()
