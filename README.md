# pubg_Scan — PUBG 纯视觉人物识别

> **平台约定（2026-09-29）：今后直接在 Windows 11 x64 + RTX 5070 上开发、测试、训练和打包，不再依赖 Mac。旧 Mac/MPS 结果仅是历史实验记录。**

人物视觉实验室 0.2.1：导入游戏图片/录像、标注自身/队友/对手、训练本地模型，并使用 ONNX 验证人物框。最终目标是 Windows 中文安装即用软件，覆盖层只给确认的对手画框，并在 Windows 内导入录像继续训练。

**当前仍是 Python 原型。最新实验模型在置信度 0.5、匹配 IoU 0.5 的固定验证中对手命中 0/8，不能作为已验收模型；Windows 安装包和 RTX 5070 实机结果尚未完成。**

## 开始开发

1. 在 Windows 克隆 `https://github.com/misitesun/pubg_Scan.git`。
2. 阅读 [Windows 开发、测试、训练指南](docs/Windows开发测试训练指南.md)，创建 Windows 本机环境。
3. 日常源码在 [outputs/person-vision](outputs/person-vision)，项目状态和工作规则见 [AGENTS.md](AGENTS.md)。
4. 当前正式数据和模型已在仓库中，选择下列路径即可继续标注、录像验证或训练。

| 内容 | 仓库路径 |
|---|---|
| 52 张正式确认项目 | `outputs/PUBG用户确认合并项目-52张/` |
| 38 张训练 / 14 张验证 | `outputs/PUBG用户确认训练集-52张/dataset.yaml` |
| 最新四类实验模型 | `outputs/PUBG身份模型-用户确认第3轮-含队友/best.onnx` |
| 继续训练权重 | `outputs/PUBG身份模型-用户确认第3轮-含队友/best.pt` |
| 上一轮及人物检测对照 | `outputs/PUBG身份模型-用户确认第2轮/`、`outputs/PUBG首轮实验模型/` |
| 用户确认原始记录 | `outputs/PUBG队友录像-14张已确认/`、`outputs/PUBG复核结果-已确认/` 中 JSON |

类别固定为 `0=self / 1=opponent / 2=teammate / 3=unknown`。best.pt 用于继续训练，best.onnx 用于推理；这些文件均不是 Windows 安装程序。

## 文档入口

- [Windows 开发测试训练指南](docs/Windows开发测试训练指南.md)
- [当前需求与开发路线](docs/当前需求与开发路线.md)
- [原始开发文档及现行需求说明](docs/纯视觉人物画框系统开发文档.md)
- [仓库内容与素材迁移](docs/仓库内容与素材迁移.md)
- [源码使用说明](outputs/person-vision/README.md)
- [最新交付说明](outputs/PUBG身份升级交付说明.md)

原始设计涉及电视剧场景及禁止外部素材导入，已被用户后续的 PUBG、图片/录像训练需求调整；它保留作历史参考。所有项目 Markdown 均标注当前 Windows 平台约定，旧实验数据和历史成绩保留，不改写为 Windows 已验证。

## 下一阶段

在 Windows 首先核对 CUDA 与 ONNX GPU 路径、复现漏检和录像时间定位；以本机高质量原始录像为主要素材，开发自动选样、轨迹关键帧标注、批量复核。已有确认标签继续复用，新增/传播标签经用户批准后进入训练。之后补齐队标/昵称关联、镜头自身判断、独立整段录像验收与安装器。

实时输入只读可见像素，不读游戏内存、不注入 DLL、不修改游戏文件、不模拟输入。训练素材允许保存，实时推理帧不落盘或上传。当前不代表 PUBG 对局使用或反作弊兼容性已验收。
