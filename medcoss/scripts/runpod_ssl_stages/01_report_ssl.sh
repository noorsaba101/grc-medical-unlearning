#!/usr/bin/env bash
# Stage 1: report-only single-modal SSL (300 epochs).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/00_env.sh"

FINAL_FILE="${OUTPUT_ROOT}/1D_text_300/checkpoint-299.pth"
STAGE_LOG="${LOG_ROOT}/01_report_ssl.log"
output_dir="${OUTPUT_ROOT}/1D_text_300"
log_dir="${LOG_ROOT}/1D_text_300"

if [[ -f "${FINAL_FILE}" ]]; then
  echo "SKIP: 01_report_ssl already completed (${FINAL_FILE} exists)"
  exit 0
fi

if [[ ! -f "${UNI_PERCEIVER_CKPT}" ]]; then
  echo "ERROR: missing Uni-Perceiver checkpoint: ${UNI_PERCEIVER_CKPT}" >&2
  exit 1
fi
if [[ ! -d "${US_REPORT}" ]]; then
  echo "ERROR: missing report dataset directory: ${US_REPORT}" >&2
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
  echo "[01_report_ssl] Resuming from ${latest}"
fi

{
  echo "======== $(date -Iseconds) 01_report_ssl start ========"
  ${DIST_LAUNCH} --master_port='29502' main_pretrain_single_modal.py \
    --model "unified_vit" \
    --batch_size 128 \
    --num_workers 10 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 300 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path "${US_REPORT}" \
    --task_modality "1D_text" \
    --load_current_pretrained_weight "${UNI_PERCEIVER_CKPT}" \
    --output_dir="${output_dir}" \
    --log_dir="${log_dir}" \
    "${resume_args[@]+"${resume_args[@]}"}"
  echo "======== $(date -Iseconds) 01_report_ssl end ========"
} 2>&1 | tee -a "${STAGE_LOG}"

if [[ ! -f "${FINAL_FILE}" ]]; then
  echo "ERROR: expected final checkpoint missing: ${FINAL_FILE}" >&2
  exit 1
fi
