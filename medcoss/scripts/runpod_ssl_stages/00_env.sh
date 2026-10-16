#!/usr/bin/env bash
# Common environment for inherited RunPod SSL stage scripts.
set -euo pipefail

RUNPOD_SSL_STAGES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MEDCOSS_ROOT="$(cd "${RUNPOD_SSL_STAGES_DIR}/../.." && pwd)"
# Preserve the variable used by the inherited per-stage scripts.
export REPO_ROOT="${MEDCOSS_ROOT}"

export DATA_ROOT="${DATA_ROOT:-/tmp/data}"
export US_REPORT="${DATA_ROOT}/us_report"
export US_XRAY="${DATA_ROOT}/us_xray"
export US_PATHOLOGY="${DATA_ROOT}/us_pathology"

export OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/output_dir}"
export LOG_ROOT="${LOG_ROOT:-/workspace/logs}"

export UNI_PERCEIVER_CKPT="${UNI_PERCEIVER_CKPT:-/workspace/checkpoints/uni-perceiver-base-L12-H768-224size-torch-pretrained.pth}"

export NUM_GPUS="${NUM_GPUS:-1}"
export DIST_LAUNCH="python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS}"

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"

echo "[00_env] REPO_ROOT=${REPO_ROOT}"
echo "[00_env] NUM_GPUS=${NUM_GPUS}"
echo "[00_env] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
