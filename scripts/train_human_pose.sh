#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON_BIN:-/home/juyiang/data/conda_envs/Utonia/bin/python}"
CONFIG="${ROOT}/configs/human_pose/hmr-utonia-v1m1-0-bedlam.py"

export CUDA_VISIBLE_DEVICES="${GPU_IDS:-0,1,2,3}"
IFS=',' read -r -a GPU_LIST <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${NUM_GPUS:-${#GPU_LIST[@]}}"
BATCH_SIZE_VAL="${BATCH_SIZE_VAL:-$((64 * NUM_GPUS))}"
export PYTHONPATH="${ROOT}"

export UTONIA_CKPT="${UTONIA_CKPT:-/home/juyiang/data/RGBD_human_pose/RGBD_HMR/utonia.pth}"
export BEDLAM_PC_ROOT="${BEDLAM_PC_ROOT:-/home/juyiang/data/bedlam/pc_row_shard}"
export BEDLAM_GT_ROOT="${BEDLAM_GT_ROOT:-/home/juyiang/data/bedlam/gt_shard}"
export BEHAVE_MANIFEST="${BEHAVE_MANIFEST:-/home/juyiang/data/PointTransformerV3_HumanPointCloud/data/behave/behave_evaluation_full_shard}"
export SMPLX_MODELS_DIR="${SMPLX_MODELS_DIR:-/home/juyiang/data/SMPL_models/models}"

SAVE_PATH="${SAVE_PATH:-${ROOT}/exp/human_pose/hmr-utonia-v1m1-0-bedlam}"
SWANLAB_PROJECT="${SWANLAB_PROJECT:-utonia_pointcept}"
SWANLAB_MODE="${SWANLAB_MODE:-online}"
BUILD_OPS="${BUILD_OPS:-0}"
RESUME="${RESUME:-false}"
WEIGHT="${WEIGHT:-${SAVE_PATH}/model/model_last.pth}"

if [[ "${RESUME}" == "true" && ( "${SWANLAB_MODE}" == "online" || "${SWANLAB_MODE}" == "cloud" ) ]]; then
  export SWANLAB_RESUME="must"
else
  unset SWANLAB_RUN_ID SWANLAB_RESUME
fi

if [[ ! -x "${PYTHON}" ]]; then
  echo "Python interpreter not found or not executable: ${PYTHON}" >&2
  exit 1
fi

if [[ "${BUILD_OPS}" != "0" ]]; then
  PYTHON_BIN="${PYTHON}" bash "${ROOT}/tools/build_human_pose_ops.sh"
fi

mkdir -p "${SAVE_PATH}"
ulimit -n 65536

exec "${PYTHON}" -u "${ROOT}/tools/train.py" \
  --config-file "${CONFIG}" \
  --num-gpus "${NUM_GPUS}" \
  --options \
    save_path="${SAVE_PATH}" \
    weight="${WEIGHT}" \
    batch_size_val="${BATCH_SIZE_VAL}" \
    resume="${RESUME}" \
    swanlab_project="${SWANLAB_PROJECT}" \
    swanlab_mode="${SWANLAB_MODE}"
