#!/bin/bash
# Đường dẫn được tính tương đối theo vị trí của script này (ai-core/bash/),
# nên luôn đúng dù script được gọi từ đâu (repo root, ai-core/, hay bash/).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

python "$ROOT_DIR/src/dataset/detection.py"
python "$ROOT_DIR/src/dataset/classification.py"
