#!/usr/bin/env bash
# Stage 2: report buffer (k-means). Reruns are OK; no --resume.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/00_env.sh"

FINAL_FILE="${OUTPUT_ROOT}/1D_text_300/1D_text_0.01_0.05_kmean.csv"
STAGE_LOG="${LOG_ROOT}/02_report_buffer.log"
REPORT_CKPT="${OUTPUT_ROOT}/1D_text_300/checkpoint-299.pth"

if [[ -f "${FINAL_FILE}" ]]; then
  echo "SKIP: 02_report_buffer already completed (${FINAL_FILE} exists)"
  exit 0
fi

if [[ ! -f "${REPORT_CKPT}" ]]; then
  echo "ERROR: missing report SSL checkpoint (run 01 first): ${REPORT_CKPT}" >&2
  exit 1
fi
if [[ ! -d "${US_REPORT}" ]]; then
  echo "ERROR: missing report dataset directory: ${US_REPORT}" >&2
  exit 1
fi
if [[ ! -f "${US_REPORT}/master.csv" ]]; then
  echo "ERROR: missing ${US_REPORT}/master.csv (required for report buffer)" >&2
  exit 1
fi

cd "${REPO_ROOT}"

{
  echo "======== $(date -Iseconds) 02_report_buffer start ========"
  python main_buffer_kmean.py \
    --model "unified_vit" \
    --num_workers 10 \
    --norm_pix_loss \
    --data_path "${US_REPORT}" \
    --task_modality "1D_text" \
    --load_current_pretrained_weight "${REPORT_CKPT}" \
    --num_center 0.01 \
    --buffer_ratio 0.05 \
    --exp_name "kmean"
  echo "======== $(date -Iseconds) 02_report_buffer end ========"
} 2>&1 | tee -a "${STAGE_LOG}"

if [[ ! -f "${FINAL_FILE}" ]]; then
  echo "ERROR: expected buffer artifact missing: ${FINAL_FILE}" >&2
  exit 1
fi
