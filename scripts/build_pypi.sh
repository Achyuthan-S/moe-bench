#!/bin/bash
# Build sdist/wheel locally. Publish: twine upload dist/*  (requires PyPI token)
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pip install build twine
python -m build
echo "Built dist/. Test: pip install dist/moe_bench-*.whl"
