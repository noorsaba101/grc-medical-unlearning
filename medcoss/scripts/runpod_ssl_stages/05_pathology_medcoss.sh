#!/usr/bin/env bash
# Stage 5: pathology MedCoSS continual (300 epochs).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/00_env.sh"

FINAL_FILE="${OUTPUT_ROOT}/MedCoSS_Report_Xray_Path_buff_0.05_cen_0.01_2D_Path_300/checkpoint-299.pth"
STAGE_LOG="${LOG_ROOT}/05_pathology_medcoss.log"
output_dir="${OUTPUT_ROOT}/MedCoSS_Report_Xray_Path_buff_0.05_cen_0.01_2D_Path_300"
log_dir="${LOG_ROOT}/MedCoSS_Report_Xray_Path_buff_0.05_cen_0.01_2D_Path_300"
XRAY_CKPT="${OUTPUT_ROOT}/MedCoSS_Report_Xray_Path_buff_0.05_cen_0.01_2D_Xray_300/checkpoint-299.pth"

if [[ -f "${FINAL_FILE}" ]]; then
  echo "SKIP: 05_pathology_medcoss already completed (${FINAL_FILE} exists)"
  exit 0
fi

if [[ ! -f "${XRAY_CKPT}" ]]; then
  echo "ERROR: missing X-ray MedCoSS checkpoint: ${XRAY_CKPT}" >&2
  exit 1
fi
if [[ ! -d "${US_REPORT}" ]] || [[ ! -d "${US_XRAY}" ]] || [[ ! -d "${US_PATHOLOGY}" ]]; then
  echo "ERROR: missing dataset dirs US_REPORT=${US_REPORT} US_XRAY=${US_XRAY} US_PATHOLOGY=${US_PATHOLOGY}" >&2
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
  echo "[05_pathology_medcoss] Resuming from ${latest}"
fi

{
  echo "======== $(date -Iseconds) 05_pathology_medcoss start ========"
  ${DIST_LAUNCH} --master_port='29361' main_pretrain_medcoss.py \
    --model "unified_vit" \
    --batch_size 128 \
    --num_workers 10 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 300 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --task_modality "2D_path" \
    --load_current_pretrained_weight "${XRAY_CKPT}" \
    --data_path_1D_text "${US_REPORT}" \
    --data_path_2D_xray "${US_XRAY}" \
    --data_path_2D_path "${US_PATHOLOGY}" \
    --output_dir="${output_dir}" \
    --log_dir="${log_dir}" \
    --num_center 0.01 \
    --buffer_ratio 0.05 \
    --exp_name "kmean" \
    --mix_up 1 \
    "${resume_args[@]+"${resume_args[@]}"}"
  echo "======== $(date -Iseconds) 05_pathology_medcoss end ========"
} 2>&1 | tee -a "${STAGE_LOG}"

if [[ ! -f "${FINAL_FILE}" ]]; then
  echo "ERROR: expected final checkpoint missing: ${FINAL_FILE}" >&2
  exit 1
fi
