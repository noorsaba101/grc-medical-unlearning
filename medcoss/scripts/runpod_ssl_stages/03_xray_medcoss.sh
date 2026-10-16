#!/usr/bin/env bash
# Stage 3: X-ray MedCoSS continual (300 epochs).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/00_env.sh"

FINAL_FILE="${OUTPUT_ROOT}/MedCoSS_Report_Xray_Path_buff_0.05_cen_0.01_2D_Xray_300/checkpoint-299.pth"
STAGE_LOG="${LOG_ROOT}/03_xray_medcoss.log"
output_dir="${OUTPUT_ROOT}/MedCoSS_Report_Xray_Path_buff_0.05_cen_0.01_2D_Xray_300"
log_dir="${LOG_ROOT}/MedCoSS_Report_Xray_Path_buff_0.05_cen_0.01_2D_Xray_300"
REPORT_CKPT="${OUTPUT_ROOT}/1D_text_300/checkpoint-299.pth"

if [[ -f "${FINAL_FILE}" ]]; then
  echo "SKIP: 03_xray_medcoss already completed (${FINAL_FILE} exists)"
  exit 0
fi

if [[ ! -f "${REPORT_CKPT}" ]]; then
  echo "ERROR: missing report SSL checkpoint: ${REPORT_CKPT}" >&2
  exit 1
fi
if [[ ! -d "${US_REPORT}" ]] || [[ ! -d "${US_XRAY}" ]]; then
  echo "ERROR: missing dataset dirs US_REPORT=${US_REPORT} US_XRAY=${US_XRAY}" >&2
  exit 1
fi

mkdir -p "${output_dir}" "${log_dir}"
cd "${REPO_ROOT}"

resume_args=()
shopt -s nullglob
ckpts=( "${output_dir}"/checkpoint-*.pth )
shopt -u nullglob
if (( ${#ckpts[@]} > 0 )); then
  latest=$(printf '%s\n' "${ckpts[@]}" | sort -V | tail -n1)
  resume_args=( --resume "${latest}" )
  echo "[03_xray_medcoss] Resuming from ${latest}"
fi

{
  echo "======== $(date -Iseconds) 03_xray_medcoss start ========"
  ${DIST_LAUNCH} --master_port='29361' main_pretrain_medcoss.py \
    --model "unified_vit" \
    --batch_size 128 \
    --num_workers 10 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 300 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --task_modality "2D_xray" \
    --load_current_pretrained_weight "${REPORT_CKPT}" \
    --data_path_1D_text "${US_REPORT}" \
    --data_path_2D_xray "${US_XRAY}" \
    --output_dir="${output_dir}" \
    --log_dir="${log_dir}" \
    --num_center 0.01 \
    --buffer_ratio 0.05 \
    --exp_name "kmean" \
    --mix_up 1 \
    "${resume_args[@]+"${resume_args[@]}"}"
  echo "======== $(date -Iseconds) 03_xray_medcoss end ========"
} 2>&1 | tee -a "${STAGE_LOG}"

if [[ ! -f "${FINAL_FILE}" ]]; then
  echo "ERROR: expected final checkpoint missing: ${FINAL_FILE}" >&2
  exit 1
fi
