import sys
from pathlib import Path

from PySide6.QtCore import QProcess, QTimer, Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                              QTabWidget, QLabel, QPushButton, QLineEdit, QFileDialog,
                              QMessageBox, QListWidget, QSplitter, QSpinBox, QDoubleSpinBox,
                              QComboBox, QTextEdit, QFormLayout)
from PySide6.QtWidgets import QCheckBox

from vision.dataset import Project, load_image
from vision.widgets import ImageCanvas
from vision.workers import Job, InferenceWorker
from vision.windows import Overlay, StopHotkey
from vision.identity import ROLES, ROLE_LABELS


def button(text, callback):
    widget = QPushButton(text)
    widget.clicked.connect(callback)
    return widget


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("人物视觉实验室 · 录像训练与画框验证")
        self.resize(1150, 790)
        self.project = None
        self.job = self.worker = None
        self.closing = False
        self.sample_dirty = False
        self.sample_index = -1
        self.training_generation = 0
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.training_output)
        self.process.finished.connect(self.training_finished)
        self.process.errorOccurred.connect(lambda error: self.training_log.append("训练进程启动失败，请检查 Python 和训练依赖。"))
        self.overlay = Overlay()
        self.hotkey = StopHotkey(self.stop_inference)
        QApplication.instance().installNativeEventFilter(self.hotkey)
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        self.build_dataset_tab()
        self.build_training_tab()
        self.build_validation_tab()
        self.statusBar().showMessage("先导入录像或图片并标注所有人物；尚未加载检测模型。")
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll_inference)
        self.timer.start(33)
        QShortcut(QKeySequence("F8"), self, activated=self.stop_inference)

    def error(self, message):
        QMessageBox.warning(self, "操作未完成", str(message))

    def choose_path(self, field, kind, pattern="所有文件 (*)"):
        path = QFileDialog.getExistingDirectory(self, "选择目录") if kind == "directory" else QFileDialog.getOpenFileName(self, "选择文件", "", pattern)[0]
        if path:
            field.setText(path)

    def path_row(self, layout, title, kind, pattern="所有文件 (*)"):
        field = QLineEdit()
        row = QHBoxLayout()
        row.addWidget(field)
        row.addWidget(button("选择", lambda: self.choose_path(field, kind, pattern)))
        layout.addRow(title, row)
        return field

    def build_dataset_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(QLabel("① 身份标注　｜　拖动框住所有人物；选择身份后右键点击已有框修改身份"))
        row = QHBoxLayout()
        row.addWidget(button("新建 / 打开项目", self.open_project))
        row.addWidget(button("导入图片", self.import_images))
        row.addWidget(button("从录像抽帧", self.import_video))
        row.addWidget(QLabel("间隔（秒）"))
        self.interval = QDoubleSpinBox()
        self.interval.setRange(0.2, 60)
        self.interval.setValue(2)
        row.addWidget(self.interval)
        self.max_samples = QSpinBox()
        self.max_samples.setRange(1, 10000)
        self.max_samples.setValue(500)
        row.addWidget(QLabel("最多图片"))
        row.addWidget(self.max_samples)
        layout.addLayout(row)
        self.project_status = QLabel("未打开项目。训练素材与标签会保存在你选择的项目目录。")
        self.project_status.setWordWrap(True)
        layout.addWidget(self.project_status)
        splitter = QSplitter()
        self.samples = QListWidget()
        self.samples.currentRowChanged.connect(self.select_sample)
        splitter.addWidget(self.samples)
        self.annotation = ImageCanvas(editable=True)
        self.annotation.changed.connect(self.mark_dirty)
        splitter.addWidget(self.annotation)
        splitter.setSizes([230, 800])
        layout.addWidget(splitter, 1)
        tools = QHBoxLayout()
        self.annotation_role=QComboBox()
        for role in ROLES:self.annotation_role.addItem(ROLE_LABELS[role],role)
        self.annotation_role.setCurrentIndex(3)
        self.annotation_role.currentIndexChanged.connect(lambda i:setattr(self.annotation,'new_box_role',self.annotation_role.itemData(i)))
        tools.addWidget(self.annotation_role)
        tools.addWidget(button("撤销最后一个框", self.annotation.undo))
        tools.addWidget(button("清空框", self.annotation.clear))
        tools.addWidget(button("保存人物标注", self.save_annotation))
        tools.addWidget(button("确认当前图无人物", self.save_negative))
        tools.addWidget(button("下一张", lambda: self.samples.setCurrentRow(min(self.samples.count()-1, self.samples.currentRow()+1))))
        tools.addWidget(button("导出训练数据集", self.export_dataset))
        layout.addLayout(tools)
        identity_row=QHBoxLayout()
        identity_row.addWidget(QLabel('导出类型'))
        self.export_type=QComboBox()
        self.export_type.addItems(['四类身份（自身 / 对手 / 队友 / 不确定）','人物检测（所有身份合并为 person）'])
        identity_row.addWidget(self.export_type)
        self.confirmed_only=QCheckBox('仅导出用户确认')
        self.confirmed_only.setChecked(True)
        identity_row.addWidget(self.confirmed_only)
        self.annotation_evidence=QLineEdit()
        self.annotation_evidence.setPlaceholderText('身份依据：队标对应关系 / 前后画面 / 用户确认；不能只依据衣服或位置')
        self.annotation_evidence.textEdited.connect(self.mark_dirty)
        identity_row.addWidget(self.annotation_evidence,1)
        layout.addLayout(identity_row)
        layout.addWidget(button('导入复核页确认结果（JSON）',self.import_identity_review))
        layout.addWidget(QLabel("验证集按录像 / 图片批次分组，避免同一录像相邻帧同时进入训练集和验证集。至少准备两个素材组。"))
        self.tabs.addTab(page, "素材与标注")

    def open_project(self):
        path = QFileDialog.getExistingDirectory(self, "选择新建或已有项目目录")
        if not path:
            return
        try:
            self.flush_annotation()
            self.project = Project(path)
            self.refresh_samples()
        except Exception as exc:
            self.error(exc)

    def refresh_samples(self):
        self.sample_index = -1
        self.sample_dirty = False
        self.samples.blockSignals(True)
        self.samples.clear()
        for index, sample in enumerate(self.project.samples):
            state = f"{'用户已确认' if sample.get('user_confirmed') else '助手标注待确认' if sample.get('roles_reviewed') else '人物已标注'} {len(sample['boxes'])} 人" if sample["reviewed"] else "待标注"
            self.samples.addItem(f"{index+1:04d}　{state}")
        self.samples.blockSignals(False)
        reviewed = sum(s["reviewed"] for s in self.project.samples)
        confirmed=sum(s.get('user_confirmed',False) for s in self.project.samples)
        self.project_status.setText(f"项目：{self.project.root}　｜　图片 {len(self.project.samples)} 张，已标注 {reviewed} 张，用户确认 {confirmed} 张")
        if self.samples.count():
            self.samples.setCurrentRow(0)
        else:
            self.annotation.set_frame(None)

    def select_sample(self, index):
        if not self.project or not 0 <= index < len(self.project.samples):
            return
        try:
            self.flush_annotation()
            sample = self.project.samples[index]
            self.annotation.set_frame(load_image(self.project.image_path(sample)), sample["boxes"],sample.get('roles'),sample.get('ignored_box_indices',[]))
            self.annotation_evidence.setText(sample.get('identity_evidence',''))
            self.sample_index = index
            self.sample_dirty = False
        except Exception as exc:
            self.error(exc)

    def mark_dirty(self):
        self.sample_dirty = True

    def flush_annotation(self):
        if self.sample_dirty and self.project and self.sample_index >= 0:
            # 清空所有框需要明确确认空场景，不能自动成为负样本。
            if self.annotation.boxes:
                self.commit_annotation()
                self.sample_dirty = False
                self.update_sample_label()
            else:
                answer = QMessageBox.question(self, "空标注", "当前图片没有框，是否确认画面中没有人物？")
                if answer == QMessageBox.StandardButton.Yes:
                    self.commit_annotation()
                else:
                    self.annotation.boxes = self.project.samples[self.sample_index]["boxes"].copy()
                    self.annotation.roles=self.project.samples[self.sample_index].get('roles',['unknown']*len(self.annotation.boxes)).copy()
                    self.annotation.ignored_box_indices=set(self.project.samples[self.sample_index].get('ignored_box_indices',[]))
                self.sample_dirty = False
                self.update_sample_label()

    def update_sample_label(self):
        sample = self.project.samples[self.sample_index]
        state='用户已确认' if sample.get('user_confirmed') else '助手标注待确认' if sample.get('roles_reviewed') else '人物已标注'
        self.samples.item(self.sample_index).setText(f"{self.sample_index+1:04d}　{state} {len(sample['boxes'])} 人")

    def commit_annotation(self):
        self.project.annotate(self.sample_index,self.annotation.boxes,self.annotation.roles)
        sample=self.project.samples[self.sample_index]
        sample['identity_evidence']=self.annotation_evidence.text().strip()
        sample['reviewer']='user_UI_review'
        sample['user_confirmed']=True
        self.project.save()

    def save_annotation(self):
        if not self.project or self.sample_index < 0:
            return self.error("请先打开项目并选择图片。")
        if not self.annotation.boxes:
            return self.error("请框住所有人物；确实没有人物时，点击“确认当前图无人物”。")
        try:
            self.commit_annotation()
            self.sample_dirty = False
            self.update_sample_label()
        except Exception as exc:
            self.error(exc)

    def save_negative(self):
        if not self.project or self.sample_index < 0:
            return self.error("请先选择图片。")
        if self.annotation.boxes:
            return self.error("当前图片仍有框，请先核对画面是否确实没有人物。")
        self.project.annotate(self.sample_index, [])
        self.project.samples[self.sample_index]['user_confirmed']=True
        self.project.samples[self.sample_index]['reviewer']='user_UI_review'
        self.project.save()
        self.sample_dirty = False
        self.update_sample_label()

    def import_identity_review(self):
        if not self.project:return self.error('请先打开对应标注项目。')
        path,_=QFileDialog.getOpenFileName(self,'导入身份确认结果','','复核结果 (*.json)')
        if path:
            try:
                self.flush_annotation()
                count=self.project.import_identity_review(path)
                self.refresh_samples()
                self.statusBar().showMessage(f'已导入用户确认 {count} 张；需修正 / 待确认样本不进入正式训练。')
            except Exception as exc:self.error(exc)

    def start_job(self, operation):
        if self.job and self.job.isRunning():
            return self.error("正在处理素材，请稍候。")
        self.flush_annotation()
        self.tabs.widget(0).setEnabled(False)
        self.statusBar().showMessage("正在处理素材，请稍候…")
        self.job = Job(operation)
        self.job.completed.connect(lambda result: self.statusBar().showMessage(f"处理完成：{result}"))
        self.job.failed.connect(self.error)
        self.job.finished.connect(self.finish_job)
        self.job.start()

    def finish_job(self):
        self.tabs.widget(0).setEnabled(True)
        self.refresh_samples()
        self.job = None
        if self.closing:
            self.close()

    def import_images(self):
        if not self.project:
            return self.error("请先新建或打开项目。")
        paths, _ = QFileDialog.getOpenFileNames(self, "导入游戏截图", "", "图片 (*.jpg *.jpeg *.png *.bmp *.webp)")
        if paths:
            project = self.project
            self.start_job(lambda cancel: f"导入 {project.import_images(paths, cancel)} 张图片")

    def import_video(self):
        if not self.project:
            return self.error("请先新建或打开项目。")
        path, _ = QFileDialog.getOpenFileName(self, "从游戏录像抽帧", "", "录像 (*.mp4 *.mkv *.avi *.mov *.webm)")
        if path:
            project = self.project
            interval, limit = self.interval.value(), self.max_samples.value()
            self.start_job(lambda cancel: f"抽取 {project.extract_video(path, interval, limit, cancel)} 张图片")

    def export_dataset(self):
        if not self.project:
            return self.error("请先打开项目。")
        self.flush_annotation()
        path = QFileDialog.getExistingDirectory(self, "选择空目录保存训练数据集")
        if path:
            try:
                yaml, training, validation = self.project.export_dataset(path,identities=self.export_type.currentIndex()==0,
                    user_confirmed_only=self.confirmed_only.isChecked())
                self.training_data.setText(str(yaml))
                QMessageBox.information(self, "数据集已导出", f"训练集 {training} 张，验证集 {validation} 张。\n{yaml}")
            except Exception as exc:
                self.error(exc)

    def build_training_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(QLabel("② 本地训练　｜　支持 person 人物检测或四类身份实验模型；按数据集类别训练"))
        form = QFormLayout()
        self.training_python = self.path_row(form, "训练环境 Python", "file", "Python (*)")
        if not getattr(sys, "frozen", False):
            self.training_python.setText(sys.executable)
        self.base_model = self.path_row(form, "初始权重（YOLOv8 / YOLO11）", "file", "模型 (*.pt)")
        self.training_data = self.path_row(form, "训练数据集", "file", "配置 (*.yaml)")
        self.training_dir = self.path_row(form, "训练输出目录", "directory")
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 10000)
        self.epochs.setValue(100)
        form.addRow("训练轮数", self.epochs)
        self.train_size = QComboBox()
        self.train_size.addItems(["640", "960", "512"])
        form.addRow("模型输入尺寸", self.train_size)
        self.device = QComboBox()
        self.device.addItems(["cpu", "0", "mps"])
        self.device.setCurrentText("0" if sys.platform == "win32" else "mps")
        form.addRow("训练设备（0=CUDA，mps=Mac）", self.device)
        layout.addLayout(form)
        row = QHBoxLayout()
        self.train_button = button("开始训练并导出 ONNX", lambda: self.start_training("train"))
        row.addWidget(self.train_button)
        row.addWidget(button("验证已有权重", lambda: self.start_training("validate")))
        row.addWidget(button("仅导出 ONNX", lambda: self.start_training("export")))
        row.addWidget(button("停止训练", self.stop_training))
        layout.addLayout(row)
        self.training_log = QTextEdit()
        self.training_log.setReadOnly(True)
        self.training_log.document().setMaximumBlockCount(1500)
        layout.addWidget(self.training_log, 1)
        layout.addWidget(QLabel("训练需要单独安装训练依赖和本地初始权重；录像 / 屏幕验证仅需 ONNX。训练产物会写入所选目录。"))
        self.tabs.addTab(page, "训练与导出")

    def start_training(self, mode):
        if self.process.state() != QProcess.ProcessState.NotRunning:
            return self.error("已有训练任务正在运行。")
        for value, name in ((self.training_python.text(), "训练环境 Python"), (self.base_model.text(), "初始权重")):
            if not value or not Path(value).is_file():
                return self.error(f"请选择存在的{name}文件。")
        if mode != "export" and not Path(self.training_data.text()).is_file():
            return self.error("请选择导出的 dataset.yaml。")
        if not self.training_dir.text():
            return self.error("请选择训练输出目录。")
        script = Path(getattr(sys, "_MEIPASS", Path(__file__).parent)) / "train_model.py"
        args = ["-u", str(script), mode, "--model", str(Path(self.base_model.text()).resolve()),
                "--output", str(Path(self.training_dir.text()).resolve()), "--epochs", str(self.epochs.value()),
                "--size", self.train_size.currentText(), "--device", self.device.currentText()]
        if mode != "export":
            args += ["--data", str(Path(self.training_data.text()).resolve())]
        Path(self.training_dir.text()).mkdir(parents=True, exist_ok=True)
        self.process.setWorkingDirectory(self.training_dir.text())
        self.training_generation += 1
        self.training_log.clear()
        self.training_log.append("训练启动后，可在这里查看进度；完成后会输出 best.pt 与 best.onnx 的位置。")
        self.process.start(str(Path(self.training_python.text()).resolve()), args)

    def training_output(self):
        self.training_log.append(bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace"))

    def training_finished(self, code, status):
        self.training_output()
        self.training_log.append("\n任务完成。" if code == 0 and status == QProcess.ExitStatus.NormalExit else f"\n任务未完成，退出码 {code}，请查看上方原因。")
        if self.closing:
            self.close()

    def stop_training(self):
        if self.process.state() != QProcess.ProcessState.NotRunning:
            generation = self.training_generation
            self.process.terminate()
            QTimer.singleShot(3000, lambda: self.kill_training_if_running(generation))

    def kill_training_if_running(self, generation):
        if generation == self.training_generation and self.process.state() != QProcess.ProcessState.NotRunning:
            self.process.kill()

    def build_validation_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(QLabel("③ 画框验证　｜　录像和屏幕测试共用检测逻辑，默认使用录像"))
        form = QFormLayout()
        self.model_path = self.path_row(form, "本地检测模型", "file", "ONNX 模型 (*.onnx)")
        model_root = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).parent
        bundled_model = model_root / "models" / "person.onnx"
        if bundled_model.is_file():
            self.model_path.setText(str(bundled_model))
        self.video_path = self.path_row(form, "验证录像", "file", "录像 (*.mp4 *.mkv *.avi *.mov *.webm)")
        self.marker_directory=self.path_row(form,'队标模板目录（可选）','directory')
        self.display_policy=QComboBox()
        self.display_policy.addItems(['仅对手（身份模型 + 连续帧过滤）','所有人物 / 身份候选（仅录像复核）'])
        form.addRow('画框规则',self.display_policy)
        layout.addLayout(form)
        row = QHBoxLayout()
        self.mode = QComboBox()
        self.mode.addItems(["录像回放", "Windows 屏幕测试"])
        row.addWidget(self.mode)
        row.addWidget(QLabel("置信度"))
        self.confidence = QDoubleSpinBox()
        self.confidence.setRange(0.05, 0.95)
        self.confidence.setSingleStep(0.05)
        self.confidence.setValue(0.35)
        row.addWidget(self.confidence)
        row.addWidget(QLabel("人物类别编号"))
        self.class_id = QSpinBox()
        self.class_id.setRange(0, 999)
        row.addWidget(self.class_id)
        row.addWidget(QLabel("显示器编号"))
        self.monitor = QSpinBox()
        self.monitor.setRange(1, 16)
        row.addWidget(self.monitor)
        self.start_button = button("开始画框", self.start_inference)
        self.stop_button = button("停止（F8）", self.stop_inference)
        self.stop_button.setEnabled(False)
        row.addWidget(self.start_button)
        row.addWidget(self.stop_button)
        layout.addLayout(row)
        self.inference_status = QLabel("未运行。仅对手需身份模型；队标模板应裁剪为人物对应的队伍编号小图。")
        layout.addWidget(self.inference_status)
        identity_note=QLabel('身份模型输出仍是候选。请先用独立录像核对自身误框、对手漏检及队友误框，合格后再测试屏幕。')
        identity_note.setWordWrap(True)
        layout.addWidget(identity_note)
        self.preview = ImageCanvas()
        layout.addWidget(self.preview, 1)
        note = QLabel("屏幕测试是 Windows 验证原型，请在获准环境使用；它不会读取游戏内存或自动操作。屏幕帧不保存、不上传。")
        note.setWordWrap(True)
        layout.addWidget(note)
        self.tabs.addTab(page, "画框验证")

    def start_inference(self):
        if self.worker is not None:
            return
        if not self.model_path.text() or not Path(self.model_path.text()).is_file():
            return self.error("请先训练并导出模型，或选择兼容的本地 ONNX；没有模型不能检测人物。")
        mode = "video" if self.mode.currentIndex() == 0 else "screen"
        if mode=='screen' and self.display_policy.currentIndex()!=0:
            return self.error('屏幕测试只允许仅对手模式，所有人物复核请使用录像。')
        source = self.video_path.text() if mode == "video" else self.monitor.value()
        if mode == "video" and (not source or not Path(source).is_file()):
            return self.error("请选择验证录像。")
        if mode == "screen":
            if sys.platform != "win32":
                return self.error("屏幕测试仅支持 Windows，当前 Mac 可以进行录像验证。")
            try:
                self.hotkey.register()
                self.overlay.prepare()
            except Exception as exc:
                self.hotkey.close()
                self.overlay.hide()
                return self.error(exc)
        self.preview.set_frame(None)
        self.worker = InferenceWorker(self.model_path.text(), mode, source,
                                      self.confidence.value(), self.class_id.value(),
                                      opponents_only=self.display_policy.currentIndex()==0,
                                      marker_directory=self.marker_directory.text().strip() or None)
        self.worker.failed.connect(self.error)
        self.worker.ready.connect(lambda provider: self.inference_status.setText(f"运行中　｜　{provider}"))
        self.worker.ended.connect(self.inference_status.setText)
        self.worker.finished.connect(self.finish_inference)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.inference_status.setText("正在加载并检查模型…")
        self.worker.start()

    def stop_inference(self):
        if self.worker:
            self.worker.requestInterruption()
            self.worker.clear()
            self.inference_status.setText("正在停止…")
        self.overlay.clear()
        self.overlay.hide()
        self.hotkey.close()

    def finish_inference(self):
        # 文件结束也消费最后一个结果，然后释放像素和模型。
        if self.worker and not self.worker.isInterruptionRequested():
            self.poll_inference()
        if self.worker:
            self.worker.clear()
        self.worker = None
        self.overlay.clear()
        self.overlay.hide()
        self.hotkey.close()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        if self.closing:
            self.close()

    def poll_inference(self):
        self.overlay.update()
        if not self.worker:
            return
        item = self.worker.take_latest()
        if item is None or self.worker.isInterruptionRequested():
            return
        frame, region, boxes, milliseconds, captured = item
        if self.worker.mode == "video":
            self.preview.set_frame(frame, [(b.x1, b.y1, b.x2, b.y2) for b in boxes],[b.role for b in boxes])
        else:
            try:
                self.overlay.set_boxes(region, boxes, captured)
            except Exception as exc:
                self.stop_inference()
                self.error(exc)
        self.statusBar().showMessage(f"当前显示 {len(boxes)} 人　｜　推理 {milliseconds:.1f} ms　｜　F8 停止")

    def closeEvent(self, event):
        active = ((self.job is not None and self.job.isRunning()) or self.worker is not None or
                  self.process.state() != QProcess.ProcessState.NotRunning)
        if active:
            self.closing = True
            if self.job:
                self.job.requestInterruption()
            self.stop_inference()
            self.stop_training()
            self.statusBar().showMessage("正在释放任务，完成后自动退出…")
            event.ignore()
            return
        try:
            self.flush_annotation()
        except Exception as exc:
            self.error(exc)
            event.ignore()
            return
        self.overlay.close()
        self.hotkey.close()
        event.accept()


def apply_style(app):
    app.setApplicationName("人物视觉实验室")
    app.setStyle("Fusion")
    app.setStyleSheet("""
        QWidget { background: #172132; color: #e5edf7; font-size: 13px; }
        QMainWindow, QTabWidget::pane { border: 0; }
        QPushButton { background: #263b54; border: 1px solid #385473; padding: 8px 12px; border-radius: 5px; }
        QPushButton:hover { background: #355477; }
        QPushButton:disabled { color: #7d8a9b; background: #222d3a; }
        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox { background: #0f1928; padding: 6px; border: 1px solid #35485f; border-radius: 4px; }
        QListWidget, QTextEdit { background: #101a2a; border: 1px solid #35485f; }
        QListWidget::item { padding: 8px; }
        QListWidget::item:selected { background: #2a5264; }
        QTabBar::tab { padding: 12px 28px; background: #111c2c; }
        QTabBar::tab:selected { background: #263b54; color: #70edc5; }
        QStatusBar { color: #a8b7ca; }
    """)
def main():
    app = QApplication(sys.argv)
    apply_style(app)
    window = Window()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
