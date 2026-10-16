#!/usr/bin/env bash
set -euo pipefail

MEDCOSS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-${MEDCOSS_ROOT}/../venv}"

echo "== Updating apt =="
apt update

echo "== Installing system packages =="
apt install -y \
  libgl1 \
  libglib2.0-0 \
  git \
  curl \
  openssh-server \
  python3-pip \
  rsync \
  tree \
  tmux \
  htop \
  net-tools

echo "== Moving to MedCoSS baseline =="
cd "${MEDCOSS_ROOT}"

echo "== Checking Python =="
if [ ! -d "${VENV_DIR}" ]; then
  echo "venv not found. Creating venv with python3.8..."
  python3.8 -m venv "${VENV_DIR}"
fi

echo "== Activating venv =="
source "${VENV_DIR}/bin/activate"

echo "== Python path =="
which python
python --version

echo "== Upgrading pip/setuptools/wheel =="
python -m pip install --upgrade pip setuptools wheel

echo "== Installing Python requirements =="
python -m pip install -r requirements_runpod.txt

echo "== Downloading NLTK data =="
python - <<'PY'
import nltk
nltk.download("punkt")
nltk.download("punkt_tab")
PY

echo "== Verifying key imports =="
python - <<'PY'
mods = [
    "torch", "tensorboard", "timm", "pandas", "nltk",
    "transformers", "cv2", "pydicom", "matplotlib",
    "nibabel", "batchgenerators", "einops", "SimpleITK",
    "tensorboardX", "monai", "medpy"
]
for m in mods:
    __import__(m)
    print("OK:", m)
PY

echo "== Verifying CUDA =="
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY

echo "RunPod setup complete."
