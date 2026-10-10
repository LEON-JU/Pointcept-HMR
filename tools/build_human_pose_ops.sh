#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
export CUDAHOSTCXX="${CUDAHOSTCXX:-/usr/bin/g++}"
export CC="${CC:-/usr/bin/gcc}"
export CXX="${CXX:-/usr/bin/g++}"
export MAX_JOBS="${MAX_JOBS:-1}"
for module in pointnet2 serialization point3d_rope; do
  (cd "${ROOT}/libs/human_pose_ops/${module}" && "${PYTHON_BIN}" setup.py build_ext --build-lib "${ROOT}")
done
