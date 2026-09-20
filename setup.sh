#!/usr/bin/env bash
set -euo pipefail
LAB_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$LAB_DIR"
unset PYTHONPATH LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1
mkdir -p logs results data/raw data/prepared data/reference models
if [[ ! -x .tools/bin/uv ]]; then
  python3 -m pip install --target .tools uv==0.12.15
fi
.tools/bin/uv python install 3.11.16 --install-dir "$LAB_DIR/.python" --no-bin
if [[ ! -x .venv/bin/python ]]; then
  .tools/bin/uv venv --python .python/cpython-3.11.16-linux-x86_64-gnu/bin/python3.11 .venv
fi
LAB_REQUIREMENTS=requirements/runtime.in
if [[ -f requirements/locked.txt ]]; then
  LAB_REQUIREMENTS=requirements/locked.txt
fi
.tools/bin/uv pip install --python .venv/bin/python -r "$LAB_REQUIREMENTS" --index-url https://pypi.org/simple
.tools/bin/uv pip check --python .venv/bin/python
./lab doctor --imports
printf '\nNext: ./lab fetch\n'
