"""仅供开发/训练环境使用；运行版无需安装 torch 或 ultralytics。"""
import argparse
from pathlib import Path
import os
import shutil
import tempfile
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description="本地游戏人物训练与 ONNX 导出")
    parser.add_argument("mode", choices=["train", "export", "validate"])
    parser.add_argument("--model", required=True, help="本地 YOLOv8/YOLO11 .pt 文件")
    parser.add_argument("--data", help="导出的 dataset.yaml")
    parser.add_argument("--output", default="training-runs")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--size", type=int, choices=[512, 640, 960], default=640)
    parser.add_argument("--device", default="cpu", help="cpu / mps / 0（CUDA）")
    parser.add_argument('--lr', type=float, default=None, help='可选：指定 AdamW 初始学习率')
    args = parser.parse_args()
    if not Path(args.model).is_file() or Path(args.model).suffix.lower() != ".pt":
        parser.error("请提供已经下载好的本地 YOLOv8/YOLO11 .pt 权重，工具不会按名称自动下载。")
    if args.mode != "export" and (not args.data or not Path(args.data).is_file()):
        parser.error("训练/验证需要存在的 --data dataset.yaml。")
    if not 1 <= args.epochs <= 10000:
        parser.error("训练轮数应在 1–10000 之间。")
    config = Path(args.output).resolve() / ".training-config"
    config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config))
    Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ.setdefault("MPLCONFIGDIR", str(config / "matplotlib"))
    try:
        from ultralytics import YOLO, settings
    except ImportError:
        parser.error("缺少训练依赖，请运行 pip install -r requirements-training.txt。")
    disabled = ["sync", "wandb", "clearml", "comet", "mlflow", "neptune", "tensorboard", "dvc", "raytune"]
    settings.update({key: False for key in disabled if key in settings})
    if args.mode != "export":
        import yaml
        data = yaml.safe_load(Path(args.data).read_text(encoding="utf-8"))
        if not isinstance(data, dict) or "download" in data:
            parser.error("只支持本地数据集，不支持包含 download 脚本的配置。")
        roles=['self','opponent','teammate','unknown']
        names=data.get('names')
        identity_data=names in (dict(enumerate(roles)),roles)
        if names not in ({0:'person'},['person']) and not identity_data:
            parser.error('只支持 person 人物数据集或工具导出的四类身份数据集。')
    model = YOLO(str(Path(args.model).resolve()), task="detect")
    if model.task != "detect":
        parser.error("请选择目标检测权重，不能使用分类/姿态/分割模型。")
    if getattr(model.model, "end2end", False):
        parser.error("暂不支持端到端模型，请使用 YOLOv8 / YOLO11 检测权重。")
    if args.mode == "train":
        training_data = Path(args.data).resolve()
        # YOLO11 的分类中间通道数随类别数改变；小数据直接改成单类会丢失基线分类层。
        # 标签仍只含 0=person，保留模型现有类别结构以完整继承权重；运行端只取人物类别。
        if not identity_data and len(model.names) > 1 and model.names.get(0) == "person":
            expanded = dict(data)
            root = Path(data.get("path") or training_data.parent)
            if not root.is_absolute():
                root = training_data.parent / root
            expanded["path"] = root.resolve().as_posix()
            expanded["names"] = model.names
            training_data = config / "dataset-preserved-head.yaml"
            training_data.write_text(yaml.safe_dump(expanded, allow_unicode=True), encoding="utf-8")
            print("保留本地基线模型结构进行人物微调；推理时人物类别仍为 0。", flush=True)
        extras=dict(optimizer='AdamW',lr0=args.lr,lrf=.1) if args.lr else {}
        result = model.train(data=str(training_data), epochs=args.epochs,
                            imgsz=args.size, device=args.device, batch=4, workers=0,
                            project=str(Path(args.output).resolve()), name="identity" if identity_data else "person",
                            cache=False, plots=False, seed=42, **extras)
        best = Path(model.trainer.best)
        print(f"\n训练完成：{best}\n指标：{result.results_dict}", flush=True)
        if best.is_file():
            YOLO(str(best)).export(format="onnx", imgsz=args.size, batch=1,
                                  dynamic=False, half=False, nms=False, simplify=False,
                                  opset=17, device="cpu")
            print(f"可导入模型：{best.with_suffix('.onnx')}", flush=True)
    elif args.mode == "export":
        output = Path(args.output).resolve()
        output.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output, prefix="export-") as temporary:
            weight = Path(temporary) / "person.pt"
            model.save(weight)
            exported = YOLO(str(weight)).export(format="onnx", imgsz=args.size, batch=1, dynamic=False,
                                                half=False, nms=False, simplify=False, opset=17, device="cpu")
            target = output / ("person-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f") + ".onnx")
            shutil.copy2(exported, target)
            print(f"可导入模型：{target}", flush=True)
    else:
        results = model.val(data=str(Path(args.data).resolve()), imgsz=args.size,
                            device=args.device, plots=False, save=False,
                            project=str(Path(args.output).resolve()), name="validation")
        print(results.results_dict, flush=True)


if __name__ == "__main__":
    main()
