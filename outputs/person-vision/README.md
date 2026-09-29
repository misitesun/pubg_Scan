# 人物视觉实验室 0.2.1

> **平台约定（2026-09-29）：今后直接在 Windows 11 x64 + RTX 5070 上开发、测试、训练和打包，不再依赖 Mac。旧 Mac/MPS 结果仅是历史实验记录。**

这是用于游戏录像标注、人物模型训练与画框验证的 Python 原型。支持原有 `person` 人物模型和新四类身份实验模型：`0=self / 1=opponent / 2=teammate / 3=unknown`。身份预测仍是实验能力，不能把模型分类置信度理解为已确认敌我身份。

## 0.2 身份标注与过滤

- 标注时先选择「自身 / 对手 / 队友 / 不确定」，拖动新增框；右键点击已有框修改身份。框住所有人物，包括自身和队友，并填写身份判断依据。未知身份保留为人物，不能清空成无人背景。
- 导出可选四类身份或单类人物数据集。旧项目的人物标注不会自动升级为身份标签，须重新复核；未复核身份的样本不会混入身份训练。
- 默认勾选「仅导出用户确认」。助手标注与用户确认分别记录，助手建议不会自动成为正式训练标签。离线复核页导出的 JSON 可通过「导入复核页确认结果」载入对应项目；工具核对原图哈希及人物框，拒绝覆盖已变化的标注。需修正和待确认样本不进入正式数据集；取消勾选只用于明确的小样本实验。
- 训练页按 dataset.yaml 类别自动训练，四类模型导出带正确 names 元数据的 ONNX。person 微调保留原分类头；身份训练使用新分类头。
- 默认「仅对手」需要身份模型，程序自动解码四类，无须填写对手类别编号。person 模型请选择「所有人物 / 身份候选（仅录像复核）」。屏幕模式只允许「仅对手」。
- 「仅对手」要求候选置信度至少 0.5、连续至少 3 次出现，且当前轨迹从未被判为自身或队友。明显切镜、轨迹失配或超过 0.5 秒的遮挡会重新确认身份。此短时 IoU 跟踪会在快速转向和遮挡时失去目标，不宣称长期稳定跟踪。
- 可选队标模板目录：提供已确认的 4–128 像素 PNG/JPG 队伍编号小图，只在人物头部附近匹配。多个身体对应同一标记时保持 unknown。不能导入衣服图片当作队标，也不能把 HUD 名单、小地图标记误当作场景人物的队标。
- 队标模板优先排除队友；仅模型预测 teammate 而缺少队标时保持 unknown。模板只能匹配已有图形和近似尺寸，不是任意队友姓名 OCR；当前 6 个正式队友训练实例来自同一段录像，尚无独立队友验证；队友排除未验收。
- 四类模型从图像学习身份候选，仍可能学习衣服、位置等错误相关性；短时稳定并不证明身份真实。当前仅用于离线录像实验，须补充多种穿搭、左右肩切换、第一 / 第三人称和真实队友片段后独立验收。
- 新身份 ONNX 必须含 `{0:'self',1:'opponent',2:'teammate',3:'unknown'}` 的 names 元数据。程序不会将 COCO 的类别 1 当作对手。

用户复核反馈可将明确死亡的实例单独标记为忽略：项目保留原身体框、身份与备注便于追溯，但导出训练标签时剔除该实例；画布显示灰色“忽略（已死亡）”。尚存活的倒地人物仍可标为对手。

队伍编号的用户参考为1黄、2橙、3蓝、4绿，须与场景身体对应的头顶队标和昵称、左下队伍名单核对；没有对应队标的非自身人物按用户当前素材约定标为对手。墙后孤立队标不能推测身体框。当前运行代码仍为局部模板实验，尚未自动读取完整名单/昵称，这次规则整理不等于队友识别已验收。

旧版下文的单类 person 操作流程仍适用于「人物检测」导出及「所有人物」录像复核。

默认使用本地录像。Windows 屏幕测试适配器已经保留，和录像共用同一个 ONNX 检测器；不需要训练完成后重写推理逻辑。已获取用户提供的视频所属合集的 5 条 PUBG 录像，制作 681 张抽帧项目，并对首批核对的 45 张图片进行小规模微调。实验模型仅用于录像验证，仍存在明显漏检，不承诺识别准确度，也不代表 Windows 游戏实测验收。

## 已实现

- 中文界面：素材与标注、训练与导出、画框验证。
- 从录像按间隔抽帧，或导入截图。鼠标拖动标注人物框，支持撤销、保存、明确确认无人物图片。
- 仅导出人工确认的标签。按录像 / 图片导入批次分组划分训练与验证集，避免同一录像相邻帧泄漏。
- 调用独立训练环境训练 YOLOv8 / YOLO11 人物或四类身份检测器，完成后导出 ONNX；支持验证已有权重和单独导出。
- 录像按原始时间回放画框，慢推理时跳过旧帧，不堆积推理结果。
- Windows 屏幕模式：系统屏幕采集，独立透明置顶鼠标穿透窗口，F8 停止，旧框过期清除。
- 推理路径没有截图、视频或检测历史输出，也没有网络客户端。
- Windows 便携应用打包脚本及 Inno Setup 安装脚本。

## 当前边界

本版本对应开发文档里的「Python 模型验证阶段」，不是最终 C++ / DirectX / TensorRT 高刷新交付版。屏幕采集使用 MSS，推理最高循环频率为 30，Windows 推理优先 DirectML；Mac ONNX Runtime CPU 仅为历史测试环境；没有 ByteTrack、运动预测、GPU 零拷贝或 144Hz / 300Hz 性能保证。

Windows 的多显示器、125% / 150% DPI、鼠标穿透、F8 全局热键、覆盖层采集排除和实际 GPU 速度，需要在 Windows 11 实机验收。Mac 上的界面与核心测试不能替代 Windows 验收。

纯视觉不意味着 PUBG 允许在对局中使用。屏幕模式不会读取游戏内存、注入 DLL、修改游戏文件、模拟鼠标键盘或规避反作弊，但不能保证不会受到游戏处罚。请先在游戏关闭时，用录制好的录像完成模型验证；实际对局用途需获得运营方明确许可。录像验证与屏幕测试不自动启动游戏，也不自动开启屏幕采集。

## 在 Windows 开发环境运行

后续开发、测试、训练和打包全部在 Windows 11 x64 + RTX 5070 完成，不再依赖 Mac。详细步骤见 [Windows 平台指南](../../docs/Windows开发测试训练指南.md)。以下命令在仓库根目录执行：

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r outputs/person-vision/requirements-training.txt
.venv\Scripts\python.exe outputs/person-vision/app.py
```

RTX 5070 训练应先按 PyTorch 官方说明安装兼容的 CUDA 版 PyTorch，再实测 CUDA 可用和显卡计算后选择设备 `0`。默认 pip 安装结果不能代替 GPU 验证。ONNX/DirectML 推理无需 PyTorch；训练 GUI 选择 Windows 本机 `.venv\Scripts\python.exe`。

## 推荐操作顺序

1. 新建项目目录。
2. 从至少两段不同录像抽帧。图片也可以分批导入；同一批图片保持为同一个素材组。不要把同一录像的相邻帧人为分成多个组。
3. 框住每一张图中的所有人物。对于遮挡人物，框出可见部分并保持统一标准；对于确定没有人物的图片，点击「确认当前图无人物」。没有标注的图片不会自动当成负样本。
4. 选择空目录导出数据集。工具生成 `dataset.yaml`、YOLO 标签和 `split.json`。
5. 在训练页选择安装好训练依赖的 Python、本地初始 `.pt` 权重、数据集和输出目录。推荐初始权重 `yolo11n.pt` 或 `yolo11s.pt`。权重需要你手动提供，不会通过填写模型名称自动下载。只加载可信权重。
6. 点击开始训练。输出会生成新的 `person` / `identity` 或递增编号目录，并显示最终 `best.pt` / `best.onnx` 路径。`best.pt` 用于继续训练，`best.onnx` 给运行软件推理使用。
7. 在画框验证页选择 `best.onnx` 和未参与训练的录像，人物类别编号保持 `0`。检查误检、漏检、远景人物、背面、遮挡和切镜效果。
8. 需要重新训练时，补充失败场景素材，再导出到新的空目录。不要用同一段训练录像证明泛化效果。

训练图像、标签、模型、训练指标是允许保存的开发素材。屏幕推理帧不落盘；二者是不同的数据路径。运行时已调用 ONNX Runtime 的遥测禁用接口，但第三方运行库及操作系统可能产生非画面状态文件。系统分页、崩溃转储、显卡驱动等系统行为不由本工具全面控制，因此不声称「整台电脑绝对无痕」。

## 模型接口

本原型只接收 YOLOv8 / YOLO11 的 **检测** ONNX，带有 `task=detect` 元数据：

- 输入：单个 FP32 RGB 张量，`[1, 3, S, S]`，固定尺寸 `S=512 / 640 / 960`。
- 输出：单个张量 `[1, 4+C, N]`，前四项为 `cx, cy, width, height`，随后为各类别置信度。
- `dynamic=False`、`half=False`、`nms=False`、`batch=1`。
- 人物模型类别 `0=person`；四类身份模型类别见开头。兼容的 COCO 检测模型也可以先验证人物推理链路，但不代表适合 PUBG 或能够判断敌我身份。
- 使用 `0=person` 的多类别本地基线权重时，训练工具会保留原分类层结构，以免小数据微调丢失预训练分类层。训练标签仍只含人物，运行端仍只显示人物类别；导出的 ONNX 可能保留其他类别通道。
- 不支持 YOLO26 的端到端输出、YOLOv5 objectness 输出、姿态 / 分割 / 分类模型、FP16 / TensorRT engine。

命令行也可使用：

```powershell
python train_model.py train --model C:\Models\best.pt --data C:\Data\dataset.yaml --output C:\Training\round4 --epochs 100 --size 960 --device 0
python train_model.py export --model C:\Models\best.pt --output C:\Models\exported --size 960
python train_model.py validate --model C:\Models\best.pt --data C:\Data\dataset.yaml --device 0
```

## Windows 打包

后续直接在 Windows x64 构建电脑打包，无 Mac 构建步骤。普通使用者最终无需安装 Python；当前为开发原型。以下命令在源码目录执行，或按平台指南使用仓库根目录路径。

```powershell
powershell -ExecutionPolicy Bypass -File packaging/build-windows.ps1 -Model C:\Models\best.onnx
```

这里的 PowerShell 参数只允许执行本地打包脚本，不涉及游戏或反作弊设置。

生成 `dist/PersonVision/PersonVision.exe` 及所需 DLL；模型可选打包到 `models/person.onnx`（兼容文件名，实际类别由元数据决定）。验证整份目录在无 Python 的干净 Windows 11 电脑上可运行后，用 Inno Setup 6 编译 `packaging/installer.iss`，得到 `PersonVision-Setup-0.2.1.exe`。安装无需管理员权限，不安装驱动，不修改系统或游戏配置。当前身份实验模型未通过精度验收，不作为默认安装即用模型打包。

安装包只包含日常标注 / 推理工具；训练页需要开发者另行配置训练 Python。如果要给普通用户交付安装即用的推理工具，必须随包提供已经训练并验收的 ONNX 模型。首版不包含自动训练环境安装或 TensorRT 编译。

## 官方资料

- [Ultralytics 训练](https://docs.ultralytics.com/modes/train/)
- [Ultralytics ONNX 导出](https://docs.ultralytics.com/modes/export/)
- [PyInstaller 平台限制](https://pyinstaller.org/en/stable/)
- [PUBG 对第三方辅助程序的说明](https://www.pubg.com/en/news/10338)
- [PyTorch 安装与 CUDA 选择](https://pytorch.org/get-started/locally/)
