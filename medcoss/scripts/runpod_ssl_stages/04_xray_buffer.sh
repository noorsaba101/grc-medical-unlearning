#!/usr/bin/env bash
# Stage 4: X-ray buffer (k-means). Reruns are OK; no --resume.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/00_env.sh"

FINAL_FILE="${OUTPUT_ROOT}/MedCoSS_Report_Xray_Path_buff_0.05_cen_0.01_2D_Xray_300/2D_xray_0.01_0.05_kmean.json"
STAGE_LOG="${LOG_ROOT}/04_xray_buffer.log"
XRAY_CKPT="${OUTPUT_ROOT}/MedCoSS_Report_Xray_Path_buff_0.05_cen_0.01_2D_Xray_300/checkpoint-299.pth"

if [[ -f "${FINAL_FILE}" ]]; then
  echo "SKIP: 04_xray_buffer already completed (${FINAL_FILE} exists)"
  exit 0
fi

if [[ ! -f "${XRAY_CKPT}" ]]; then
  echo "ERROR: missing X-ray MedCoSS checkpoint (run 03 first): ${XRAY_CKPT}" >&2
  exit 1
fi
if [[ ! -d "${US_XRAY}" ]]; then
  echo "ERROR: missing X-ray dataset directory: ${US_XRAY}" >&2
  exit 1
fi

cd "${REPO_ROOT}"

{
  echo "======== $(date -Iseconds) 04_xray_buffer start ========"
  python main_buffer_kmean.py \
    --model "unified_vit" \
    --num_workers 10 \
    --norm_pix_loss \
    --data_path "${US_XRAY}" \
    --task_modality "2D_xray" \
    --load_current_pretrained_weight "${XRAY_CKPT}" \
    --num_center 0.01 \
    --buffer_ratio 0.05 \
    --exp_name "kmean"
  echo "======== $(date -Iseconds) 04_xray_buffer end ========"
} 2>&1 | tee -a "${STAGE_LOG}"

if [[ ! -f "${FINAL_FILE}" ]]; then
  echo "ERROR: expected buffer artifact missing: ${FINAL_FILE}" >&2
  exit 1
fi
