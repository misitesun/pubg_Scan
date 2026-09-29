import threading
import time
from PySide6.QtCore import QThread, Signal

from .detector import Detector
from .sources import ScreenSource, VideoSource


class Job(QThread):
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, operation):
        super().__init__()
        self.operation = operation

    def run(self):
        try:
            result = self.operation(self.isInterruptionRequested)
            self.completed.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class InferenceWorker(QThread):
    failed = Signal(str)
    ready = Signal(str)
    ended = Signal(str)

    def __init__(self, model, mode, source, threshold, class_id, opponents_only=False, marker_directory=None):
        super().__init__()
        self.model, self.mode, self.source = model, mode, source
        self.threshold, self.class_id = threshold, class_id
        self.lock = threading.Lock()
        self.latest = None
        self.opponents_only=opponents_only
        self.marker_directory=marker_directory
        self.decisions=[]

    def take_latest(self):
        with self.lock:
            result, self.latest = self.latest, None
        return result

    def run(self):
        source = None
        outcome = "已停止"
        try:
            detector = Detector(self.model, threshold=self.threshold, class_id=self.class_id)
            from .identity import IdentityFilter,TeamMarkers
            identity=IdentityFilter(TeamMarkers(self.marker_directory)) if self.opponents_only else None
            if self.opponents_only and not detector.identity_model:
                raise ValueError('只框对手需要四类身份模型；当前 person 模型只能用于人物检测复核。')
            if self.mode=='screen' and not self.opponents_only:
                raise ValueError('屏幕测试仅允许只显示对手；所有人物复核仅用于录像。')
            if self.isInterruptionRequested():
                return
            source = VideoSource(self.source) if self.mode == "video" else ScreenSource(self.source)
            self.ready.emit(detector.provider)
            while not self.isInterruptionRequested():
                started = time.perf_counter()
                item = source.read()
                if item is None:
                    outcome = "录像播放结束"
                    break
                frame, region, captured = item
                boxes, milliseconds = detector.detect(frame)
                if identity is not None:
                    boxes=identity.update(frame,boxes,captured)
                    self.decisions=identity.decisions
                if self.isInterruptionRequested():
                    break
                # 单槽覆盖，无逐帧 signal 排队；屏幕模式不把像素传入界面。
                with self.lock:
                    self.latest = (frame if self.mode == "video" else None,
                                   region, boxes, milliseconds, captured)
                if self.mode == "screen":
                    del frame
                fps_limit = min(source.fps, 30) if self.mode == "video" else 30
                wait = max(0, 1 / fps_limit - (time.perf_counter() - started))
                self.msleep(round(wait * 1000))
        except Exception as exc:
            outcome = "运行失败"
            self.failed.emit(str(exc))
        finally:
            if source is not None:
                source.close()
            self.ended.emit(outcome)

    def clear(self):
        with self.lock:
            self.latest = None
            self.decisions=[]
