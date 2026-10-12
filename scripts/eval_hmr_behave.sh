#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON_BIN:-/home/juyiang/data/conda_envs/Utonia/bin/python}"

export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export EVAL_CONFIG="${EVAL_CONFIG:-${ROOT}/exp/human_pose/hmr-utonia-v1m1-0-bedlam-256/config.py}"
export EVAL_WEIGHT="${EVAL_WEIGHT:-${ROOT}/exp/human_pose/hmr-utonia-v1m1-0-bedlam-256/model/model_best.pth}"
export BEHAVE_EVAL_SEQ_ROOT="${BEHAVE_EVAL_SEQ_ROOT:-/home/juyiang/data/PointTransformerV3_HumanPointCloud/data/behave/behave_evaluation_full_shard}"
export SMPL_MODELS_DIR="${SMPL_MODELS_DIR:-/home/juyiang/data/SMPL_models/models}"
export EVAL_DEVICE="${EVAL_DEVICE:-cuda:0}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
export EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-4}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Python interpreter not found or not executable: ${PYTHON}" >&2
  exit 1
fi

exec "${PYTHON}" -u "${ROOT}/tools/eval_hmr_behave.py" "$@"
