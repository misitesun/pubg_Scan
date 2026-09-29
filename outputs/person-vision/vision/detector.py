"""录像与屏幕共用的 ONNX 推理；无网络、无图像文件输出。"""
from dataclasses import dataclass
import ast
from pathlib import Path
import time

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    role: str = 'unknown'
    class_id: int = 0


def letterbox(frame, size):
    height, width = frame.shape[:2]
    scale = min(size / width, size / height)
    resized_width, resized_height = round(width * scale), round(height * scale)
    left, top = (size - resized_width) // 2, (size - resized_height) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[top:top + resized_height, left:left + resized_width] = cv2.resize(
        frame, (resized_width, resized_height)
    )
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255
    return tensor, scale, left, top


def decode_yolo(raw, frame_shape, scale, left, top, threshold, iou, class_id):
    """仅接受 YOLOv8/YOLO11 原始检测输出 [1, 4+类别数, 候选数]。"""
    raw = np.asarray(raw)
    if raw.ndim != 3 or raw.shape[0] != 1 or raw.shape[1] < 5 or raw.shape[2] < 100:
        raise ValueError("模型输出不兼容：请导入工具导出的 YOLOv8/YOLO11 检测 ONNX（nms=False）。")
    if not 0 <= class_id < raw.shape[1] - 4:
        raise ValueError("人物类别编号超出模型类别范围。")
    rows = raw[0].T
    scores = rows[:, 4 + class_id]
    valid = (np.isfinite(rows).all(axis=1) & (scores >= threshold) &
             (np.argmax(rows[:, 4:], axis=1) == class_id))
    rows, scores = rows[valid], scores[valid]
    if not len(rows):
        return []
    height, width = frame_shape[:2]
    boxes = rows[:, :4].copy()
    boxes[:, 0] = (rows[:, 0] - rows[:, 2] / 2 - left) / scale
    boxes[:, 1] = (rows[:, 1] - rows[:, 3] / 2 - top) / scale
    boxes[:, 2] = (rows[:, 0] + rows[:, 2] / 2 - left) / scale
    boxes[:, 3] = (rows[:, 1] + rows[:, 3] / 2 - top) / scale
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height)
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes, scores = boxes[valid], scores[valid]
    xywh = np.column_stack((boxes[:, :2], boxes[:, 2:] - boxes[:, :2]))
    # NMS 不限制远景小人物的面积。
    selected = cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), threshold, iou)
    return [Detection(*map(float, boxes[int(i)]), float(scores[int(i)]))
            for i in np.asarray(selected).reshape(-1)[:128]]


class Detector:
    def __init__(self, model_path, threshold=0.35, iou=0.55, class_id=0):
        import onnxruntime as ort
        ort.disable_telemetry_events()
        path = Path(model_path)
        if not path.is_file() or path.suffix.lower() != ".onnx":
            raise ValueError("请选择已存在的本地 .onnx 检测模型。")
        available = ort.get_available_providers()
        providers = (["DmlExecutionProvider"] if "DmlExecutionProvider" in available else [])
        providers += ["CPUExecutionProvider"]
        options = ort.SessionOptions()
        options.log_severity_level = 3
        if "DmlExecutionProvider" in providers:
            options.enable_mem_pattern = False
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
        if self.session.get_modelmeta().custom_metadata_map.get("task") != "detect":
            raise ValueError("模型缺少 task=detect 元数据，请使用提供的训练 / 导出工具生成检测 ONNX。")
        if len(self.session.get_outputs()) != 1:
            raise ValueError("仅支持单输出的人物检测模型，不支持分割 / 姿态等模型。")
        inputs = self.session.get_inputs()
        if len(inputs) != 1:
            raise ValueError("仅支持单输入检测模型。")
        shape = inputs[0].shape
        if len(shape) != 4 or shape[:2] != [1, 3] or not isinstance(shape[2], int) or shape[2] != shape[3]:
            raise ValueError("请使用固定尺寸、batch=1、RGB NCHW 的检测模型。")
        if inputs[0].type != "tensor(float)":
            raise ValueError("原型仅支持 FP32 ONNX；请按提供的导出流程生成模型。")
        self.input_name, self.size = inputs[0].name, shape[2]
        if not 32 <= self.size <= 2048:
            raise ValueError("模型输入尺寸不在支持范围内。")
        self.provider = self.session.get_providers()[0]
        self.threshold, self.iou, self.class_id = threshold, iou, class_id
        names_text=self.session.get_modelmeta().custom_metadata_map.get('names','{}')
        try:
            names=ast.literal_eval(names_text)
            self.names={int(k):str(v) for k,v in names.items()} if isinstance(names,dict) else dict(enumerate(names))
        except (ValueError,TypeError,SyntaxError,AttributeError):
            self.names={}
        from .identity import ROLES
        self.identity_model=self.names==dict(enumerate(ROLES))
        # 导入时试推理，提前发现分类、分割、带 NMS 等不兼容模型。
        self.detect(np.zeros((64, 64, 3), dtype=np.uint8))

    def detect(self, frame):
        started = time.perf_counter()
        tensor, scale, left, top = letterbox(frame, self.size)
        outputs = self.session.run(None, {self.input_name: tensor})
        if self.identity_model:
            from dataclasses import replace
            boxes=[]
            for cid,role in self.names.items():
                boxes.extend(replace(b,role=role,class_id=cid) for b in decode_yolo(
                    outputs[0],frame.shape,scale,left,top,self.threshold,self.iou,cid))
            # 不同身份候选也去重，同一个身体只保留一个框。
            unique=[]
            from .identity import overlap
            for b in sorted(boxes,key=lambda b:b.confidence,reverse=True):
                if all(overlap(b,p)<self.iou for p in unique):unique.append(b)
            boxes=unique[:128]
        else:
            boxes = decode_yolo(outputs[0], frame.shape, scale, left, top,
                                self.threshold, self.iou, self.class_id)
        return boxes, (time.perf_counter() - started) * 1000
