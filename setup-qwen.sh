#!/usr/bin/env bash
set -euo pipefail
LAB_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$LAB_DIR"
unset PYTHONPATH LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1
# Reuse the lab's uv installation; keep the original inference environment intact.
.tools/bin/uv python install 3.12.14 --install-dir "$LAB_DIR/.python" --no-bin
if [[ ! -x .venv-qwen/bin/python ]]; then
  .tools/bin/uv venv --python .python/cpython-3.12.14-linux-x86_64-gnu/bin/python3.12 .venv-qwen
fi
QWEN_REQUIREMENTS=requirements/qwen.in
if [[ -f requirements/qwen-locked.txt ]]; then
  QWEN_REQUIREMENTS=requirements/qwen-locked.txt
fi
.tools/bin/uv pip install --python .venv-qwen/bin/python -r "$QWEN_REQUIREMENTS" --index-url https://pypi.org/simple
.tools/bin/uv pip check --python .venv-qwen/bin/python
./lab fetch --only qwen3-asr-0.6b
