param([string]$Model = "")
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
if (-not (Test-Path ".build-env/Scripts/python.exe")) {
    py -3.12 -m venv .build-env
    if ($LASTEXITCODE -ne 0) { throw "请先安装 Python 3.12 x64（仅打包电脑需要）" }
}
$PythonExe = (Resolve-Path ".build-env/Scripts/python.exe").Path
& $PythonExe -m pip install -r requirements.txt "pyinstaller>=6.11,<7"
if ($LASTEXITCODE -ne 0) { throw "依赖安装失败" }
& $PythonExe -m PyInstaller --noconfirm --clean --windowed --onedir `
    --name PersonVision --collect-all onnxruntime --collect-all mss `
    --add-data "train_model.py;." app.py
if ($LASTEXITCODE -ne 0) { throw "应用打包失败" }
Copy-Item README.md dist/PersonVision/使用说明.md
if ($Model) {
    if (-not (Test-Path $Model) -or [IO.Path]::GetExtension($Model) -ne ".onnx") {
        throw "请提供存在的 ONNX 文件"
    }
    & $PythonExe -c "from vision.detector import Detector; import sys; Detector(sys.argv[1]); print('模型验证通过')" $Model
    if ($LASTEXITCODE -ne 0) { throw "模型与原型不兼容" }
    New-Item -ItemType Directory -Force dist/PersonVision/models | Out-Null
    Copy-Item $Model dist/PersonVision/models/person.onnx
}
Write-Host "便携目录已生成：dist/PersonVision。请在 Windows 实机测试后编译 installer.iss。"
