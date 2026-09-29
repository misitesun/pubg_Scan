#!/bin/zsh
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR" || exit 1
if [[ -x "$PROJECT_DIR/.venv/bin/python" ]]; then
    "$PROJECT_DIR/.venv/bin/python" app.py
elif [[ -x "$PROJECT_DIR/../../work/.venv/bin/python" ]]; then
    "$PROJECT_DIR/../../work/.venv/bin/python" app.py
else
    echo "请先按照 README.md 配置 Python 3.12 和依赖，再启动工具。"
    read -r "?按回车退出"
fi
