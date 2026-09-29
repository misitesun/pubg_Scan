# Windows 开发、测试、训练指南

> **平台约定（2026-09-29）：今后直接在 Windows 11 x64 + RTX 5070 上开发、测试、训练和打包，不再依赖 Mac。旧 Mac/MPS 结果仅是历史实验记录。**

这是开发者接手当前 Python 原型的步骤。最终普通使用者的安装即用体验尚未完成；以下环境配置不等于最终安装包。

## 1. 获取项目与创建本机环境

安装 Git 和 Python 3.12 x64，在 PowerShell 执行；工作目录示例仅为建议，可自行选择。

```powershell
git clone https://github.com/misitesun/pubg_Scan.git
cd pubg_Scan
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r outputs/person-vision/requirements.txt
.venv\Scripts\python.exe outputs/person-vision/app.py
```

直接调用虚拟环境 Python，不要求修改 PowerShell 全局执行策略。不复制 Mac 的 work/.venv；Windows 环境在本机创建。

## 2. 配置 RTX 5070 训练环境

按 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/) 和显卡驱动兼容情况，在本项目 `.venv` 中安装支持 RTX 5070 的 CUDA 版 PyTorch；执行官方命令时将 `pip` 替换为 `.venv\Scripts\python.exe -m pip`。不要只依据依赖安装成功判断 GPU 可用。

随后安装训练依赖并实测 CUDA：

```powershell
.venv\Scripts\python.exe -m pip install -r outputs/person-vision/requirements-training.txt
.venv\Scripts\python.exe -c "import torch; print('torch:',torch.__version__); print('CUDA:',torch.version.cuda); assert torch.cuda.is_available(), 'CUDA不可用'; print(torch.cuda.get_device_name(0)); x=torch.randn(16,16,device='cuda'); print((x@x).sum().item())"
```

记录 Python、torch、CUDA、显卡驱动及 GPU 名称，保留真实结果。发生架构不兼容或 CUDA 不可用时先修复环境；不能用 CPU 跑通结果称为 RTX 5070 验证通过。现有源代码在 Windows 推理优先 DirectML，与 PyTorch CUDA 训练是两条独立路径，分别验收。

## 3. 导入现有素材与录像验证

启动后在「素材与标注」中打开 `outputs/PUBG用户确认合并项目-52张`。全部 52 张已确认，不需要为了每轮训练重复确认旧标签。新增素材和修改标签继续走用户确认流程。

在「画框验证」选择 `outputs/PUBG身份模型-用户确认第3轮-含队友/best.onnx` 和本机录像，先检验加载、时间/坐标与停止清理；当前模型漏检明显，不作合格模型使用。完整新录像必须另行导入，仓库未包含旧原始 MP4。

## 4. 在 Windows 继续训练

GUI 的训练 Python 选择仓库 `.venv\Scripts\python.exe`，模型选择最新 best.pt，数据选择 52 张数据集的 dataset.yaml，设备填 `0`，结果选择新的空目录。不要填 `mps` 或引用 Mac 绝对路径。

命令行等效示例（在仓库根目录）：

```powershell
.venv\Scripts\python.exe outputs/person-vision/train_model.py train --model "outputs/PUBG身份模型-用户确认第3轮-含队友/best.pt" --data "outputs/PUBG用户确认训练集-52张/dataset.yaml" --output "training-runs/windows-round-4" --epochs 100 --size 960 --device 0 --lr 0.001
```

这是复现流程示例，不代表再训练 100 轮即可修复漏检。先诊断、补充确认素材，再决定模型、分辨率和训练参数。保留整段录像分组和独立测试录像；6 个现有队友全部在训练集，独立队友验证仍需新增录像。

best.pt 继续训练，best.onnx 推理。旧 args.yaml 含 Mac 实际路径，仅用于历史追溯；使用可移动的 dataset.yaml 和本机新输出目录。

## 5. 测试与实机验收

```powershell
.venv\Scripts\python.exe -m unittest discover -s outputs/person-vision/tests -v
```

核心测试后记录 Windows GUI、录像推理、DirectML provider 与速度、DPI、多显示器、F8、鼠标穿透、覆盖层采集排除及长时间运行结果。先在普通可控窗口测屏幕路径，不以“纯视觉”推定对局兼容性。对手精度需统计最终 ONNX 加身份过滤的漏框/误框，不能只看训练 mAP。

## 6. 构建安装包

在 Windows 仓库根目录执行：

```powershell
powershell -ExecutionPolicy Bypass -File outputs/person-vision/packaging/build-windows.ps1
```

当前实验模型未合格，先构建开发版并在界面手动导入模型。模型通过验收后，才用 `-Model "C:\Models\best.onnx"` 将其随包内置。脚本在源码目录生成 `dist/PersonVision`；安装 Inno Setup 6 后编译 `outputs/person-vision/packaging/installer.iss`。

当前打包脚本不包含自动配置训练组件。后续还要实现图形化训练环境安装/训练版，并在无 Python 的干净 Windows 电脑验证安装、运行、训练组件、升级和卸载，才能完成最终目标。
