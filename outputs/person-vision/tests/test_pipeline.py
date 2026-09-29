"""合成素材验证工程链路，不用于宣称人物识别精度。"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
import sys
import tempfile
import time
import unittest

import cv2
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
from PySide6.QtCore import QPointF, QEventLoop, QTimer
from PySide6.QtWidgets import QApplication

from vision.dataset import Project, save_image
from vision.detector import Detector, decode_yolo, letterbox
from vision.sources import ScreenSource, VideoSource
from vision.widgets import ImageCanvas
from vision.workers import InferenceWorker


def make_contract_model(path, task="detect"):
    values = np.zeros((1, 5, 100), dtype=np.float32)
    values[0, :, 0] = [32, 32, 20, 20, 0.9]
    graph = helper.make_graph([
        helper.make_node("Constant", [], ["output"], value=numpy_helper.from_array(values))
    ], "contract_fixture", [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 64, 64])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 5, 100])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
    helper.set_model_props(model, {"task": task})
    onnx.save(model, path)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def create_video(self, name="clip.avi", frames=12):
        path = self.root / name
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 6, (96, 64))
        self.assertTrue(writer.isOpened())
        for index in range(frames):
            image = np.full((64, 96, 3), index * 10, dtype=np.uint8)
            cv2.rectangle(image, (20, 10), (40, 50), (0, 180, 90), 2)
            writer.write(image)
        writer.release()
        return path

    def test_letterbox_coordinate_roundtrip(self):
        image = np.zeros((360, 1280, 3), dtype=np.uint8)
        tensor, scale, left, top = letterbox(image, 640)
        self.assertEqual(tensor.shape, (1, 3, 640, 640))
        x1, y1, x2, y2 = 100, 30, 350, 300
        raw = np.zeros((1, 5, 100), dtype=np.float32)
        raw[0, :, 0] = [(x1+x2)/2*scale+left, (y1+y2)/2*scale+top,
                        (x2-x1)*scale, (y2-y1)*scale, 0.9]
        boxes = decode_yolo(raw, image.shape, scale, left, top, 0.35, 0.55, 0)
        self.assertEqual(len(boxes), 1)
        np.testing.assert_allclose([boxes[0].x1, boxes[0].y1, boxes[0].x2, boxes[0].y2], [x1,y1,x2,y2])

    def test_nms_invalid_values_and_other_classes(self):
        raw = np.zeros((1, 6, 100), dtype=np.float32)
        raw[0, :, 0] = [20, 20, 10, 10, .9, .1]
        raw[0, :, 1] = [20, 20, 10, 10, .8, .1]
        raw[0, :, 2] = [50, 50, 10, 10, .1, .99]
        raw[0, :, 3] = [np.nan, 50, 10, 10, .99, .1]
        raw[0, :, 4] = [75, 75, 10, 10, .6, .95]  # 其它类别更强，不应该画人物框。
        boxes = decode_yolo(raw, (100,100,3), 1,0,0,.35,.55,0)
        self.assertEqual(len(boxes), 1)
        self.assertAlmostEqual(boxes[0].confidence, .9, places=5)
        with self.assertRaises(ValueError):
            decode_yolo(np.zeros((1,100,6)), (100,100,3),1,0,0,.35,.55,0)

    def test_model_session_and_metadata_rejection(self):
        model = self.root / "contract.onnx"
        make_contract_model(model)
        detector = Detector(model)
        boxes, duration = detector.detect(np.zeros((64,64,3), dtype=np.uint8))
        self.assertEqual(len(boxes), 1)
        self.assertGreaterEqual(duration, 0)
        make_contract_model(self.root / "bad.onnx", task="pose")
        with self.assertRaises(ValueError):
            Detector(self.root / "bad.onnx")

    def test_video_extraction_duplicate_rejection_and_group_split(self):
        project = Project(self.root / "中文项目")
        first = self.create_video()
        second = self.create_video("clip2.avi", frames=10)
        self.assertEqual(project.extract_video(first, interval=1, limit=10), 2)
        self.assertEqual(project.extract_video(second, interval=1, limit=10), 2)
        with self.assertRaises(ValueError):
            project.extract_video(first)
        for index in range(4):
            project.annotate(index, [[20,10,40,50]])
        # 未确认的样本不能混入数据集。
        image = self.root / "unknown.png"
        save_image(image, np.zeros((64,96,3), dtype=np.uint8))
        project.import_images([image])
        yaml, training, validation = project.export_dataset(self.root / "数据集")
        self.assertTrue(yaml.is_file())
        self.assertEqual(training + validation, 4)
        import json
        split = json.loads((yaml.parent / "split.json").read_text())
        self.assertFalse(set(split["train_groups"]) & set(split["val_groups"]))
        label = next((yaml.parent / "labels" / "train").glob("*.txt")).read_text().split()
        np.testing.assert_allclose(list(map(float,label[1:])), [30/96,30/64,20/96,40/64])
        self.assertEqual(len(Project(project.root).samples),5)

    def test_empty_annotation_explicitly_reviewed_and_export_guard(self):
        project = Project(self.root / "project")
        image = self.root / "sample.jpg"
        save_image(image, np.zeros((64,96,3),dtype=np.uint8))
        project.import_images([image])
        self.assertFalse(project.samples[0]["reviewed"])
        project.annotate(0, [])
        self.assertTrue(project.samples[0]["reviewed"])
        with self.assertRaises(ValueError):
            project.export_dataset(self.root / "dataset")

    def test_duplicate_images_do_not_leak_between_batches(self):
        project = Project(self.root / "project")
        image = self.root / "same.jpg"
        save_image(image, np.zeros((64,96,3), dtype=np.uint8))
        self.assertEqual(project.import_images([image]), 1)
        self.assertEqual(project.import_images([image]), 0)
        self.assertEqual(len(project.samples), 1)

    def test_canvas_letterboxing_click_mapping(self):
        canvas = ImageCanvas(editable=True)
        canvas.resize(400,400)
        canvas.set_frame(np.zeros((100,200,3),dtype=np.uint8))
        # 图像垂直居中，上下留白各 100。
        point = canvas.image_point(QPointF(100,150))
        self.assertAlmostEqual(point.x(),50)
        self.assertAlmostEqual(point.y(),25)
        canvas.close()

    def test_video_source_and_non_windows_screen_guard(self):
        source = VideoSource(self.create_video())
        frame, region, timestamp = source.read()
        self.assertEqual((region.width,region.height),(96,64))
        self.assertEqual(frame.shape,(64,96,3))
        source.close()
        if sys.platform != "win32":
            with self.assertRaises(ValueError):
                ScreenSource()

    def test_real_onnx_worker_lifecycle_and_no_runtime_outputs(self):
        video = self.create_video(frames=3)
        model = self.root / "contract.onnx"
        make_contract_model(model)
        before = {p.name for p in self.root.iterdir()}
        worker = InferenceWorker(str(model), "video", str(video), .35,0)
        failures = []
        worker.failed.connect(failures.append)
        loop = QEventLoop()
        worker.finished.connect(loop.quit)
        QTimer.singleShot(5000, loop.quit)
        worker.start()
        loop.exec()
        if worker.isRunning():
            worker.requestInterruption()
            worker.wait(5000)
        self.assertFalse(worker.isRunning())
        self.assertFalse(failures)
        result = worker.take_latest()
        self.assertIsNotNone(result)
        self.assertEqual(len(result[2]),1)
        worker.clear()
        self.assertIsNone(worker.take_latest())
        self.assertEqual(before, {p.name for p in self.root.iterdir()})


if __name__ == "__main__":
    unittest.main()
