#!/usr/bin/env bash
# Run full RunPod SSL pipeline in order (resumable per-stage scripts).
set -euo pipefail

SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=00_env.sh
source "${SDIR}/00_env.sh"

cd "${REPO_ROOT}"

bash "${SDIR}/01_report_ssl.sh"
bash "${SDIR}/02_report_buffer.sh"
bash "${SDIR}/03_xray_medcoss.sh"
bash "${SDIR}/04_xray_buffer.sh"
bash "${SDIR}/05_pathology_medcoss.sh"

echo "SUCCESS: all RunPod SSL stages completed (see ${LOG_ROOT}/*.log)."
