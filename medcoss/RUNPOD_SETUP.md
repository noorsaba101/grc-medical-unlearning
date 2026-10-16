# RunPod Setup Notes for MedCoSS

## Purpose

This document records the RunPod environment, setup steps, folder structure, dependency installation, dataset layout, training workflow, and operational rules for MedCoSS training/reproduction.

---

## RunPod Configuration

Chosen configuration:

- GPU: 4× A100 SXM, 80 GB each
- Total VRAM: 320 GB
- CPU: 96 vCPU
- RAM: 468 GB
- Container image: `pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel`
- Template: `runpod-torch-v240`

Important:

The pod may have 4× A100 GPUs, but the current MedCoSS SSL scripts use **2 GPUs**:

```bash
export CUDA_VISIBLE_DEVICES=0,1
DIST_LAUNCH="torchrun --standalone --nproc_per_node=2"
```

To use 4 GPUs later, update both:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
DIST_LAUNCH="python -m torch.distributed.launch --nproc_per_node=4"
```

---

## Storage Configuration

- Volume disk: `/workspace`
- `/workspace` size: 500 GB
- Container disk: 200 GB
- Total disk: 700 GB

Important:

- `/workspace` persists when the pod is stopped and restarted.
- `/workspace` is deleted if the pod is terminated.
- Container disk is temporary and should not store important final outputs.
- `/tmp` is on container storage and is useful for faster temporary training I/O.
- Network volume was not used because the selected 4× A100 SXM configuration did not support it.

Recommended strategy:

- Keep persistent copies of code, datasets, checkpoints, and important outputs under `/workspace`.
- Copy active datasets and training outputs to `/tmp` during training for faster I/O.
- After successful runs, copy `/tmp/output_dir`, `/tmp/logs`, and `/tmp/snapshots` back to `/workspace`.

---

## SSH Setup

Local SSH key generated using:

```bash
ssh-keygen -t ed25519 -C "runpod"
```

Add the public key in:

**RunPod → Settings → SSH Public Keys**

### Normal RunPod SSH

This is for terminal access only.

```bash
ssh <pod-id>@ssh.runpod.io -i ~/.ssh/id_ed25519
```

RunPod may show a command like:

```bash
ssh crw2ownj4wbdf5-64412097@ssh.runpod.io -i ~/.ssh/id_ed25519
```

Important:

- This route uses RunPod's SSH proxy.
- It does **not** support SCP/SFTP.
- Use it only for terminal access.

### SSH over Exposed TCP

This supports SSH, SCP, and SFTP.

Example:

```bash
ssh root@154.54.102.51 -p 11528 -i ~/.ssh/id_ed25519
```

On Windows CMD, use:

```bat
ssh root@154.54.102.51 -p 11528 -i %USERPROFILE%\.ssh\id_ed25519
```

Use this method for uploading files with `scp`.

General format:

```bat
scp -P <PORT> -i %USERPROFILE%\.ssh\id_ed25519 <LOCAL_FILE_PATH> root@<POD_IP>:/tmp/
```

Example upload from Windows CMD:

```bat
scp -P 11528 -i %USERPROFILE%\.ssh\id_ed25519 C:\D\4th\Thesis\data_small\data_small.tar.gz root@154.54.102.51:/tmp/
```

For multiple files:

```bat
scp -P 11528 -i %USERPROFILE%\.ssh\id_ed25519 C:\D\4th\Thesis\data\file1.tif C:\D\4th\Thesis\data\file2.tif root@154.54.102.51:/tmp/
```

---

## Start Command Fix

If the container exits immediately, set the RunPod start command to:

```bash
sleep infinity
```

---

## SSH Server Setup Inside Pod

Some containers do not have an SSH server installed/running by default. If SSH over exposed TCP gives `Connection refused`, install and start `sshd`.

Run inside the RunPod web terminal:

```bash
apt update
apt install -y openssh-server

mkdir -p /run/sshd
ssh-keygen -A
/usr/sbin/sshd
```

If SSH asks for a password or gives `Permission denied`, add the local public key to the pod.

On Windows CMD, show the public key:

```bat
type %USERPROFILE%\.ssh\id_ed25519.pub
```

Copy the full line, then inside the pod:

```bash
mkdir -p /root/.ssh
chmod 700 /root/.ssh

echo "PASTE_FULL_PUBLIC_KEY_HERE" >> /root/.ssh/authorized_keys

chmod 600 /root/.ssh/authorized_keys
```

Restart SSH server:

```bash
pkill sshd || true
/usr/sbin/sshd
```

Then retry from Windows:

```bat
ssh root@154.54.102.51 -p 11528 -i %USERPROFILE%\.ssh\id_ed25519
```

---

## Use `tmux` for Long Runs

Long SSL stages, buffer creation, and downstream training should be run inside `tmux`.

Install if needed:

```bash
apt update
apt install -y tmux
```

Start a session:

```bash
tmux new -s ssl
```

Detach safely while keeping the process running:

```text
Ctrl+b then d
```

Reattach:

```bash
tmux attach -t ssl
```

List sessions:

```bash
tmux ls
```

Kill a session:

```bash
tmux kill-session -t ssl
```

This is important because normal SSH disconnection can kill the running Python process if it is not inside `tmux`.

---

## File Upload and Extraction

### Upload Small Dataset Using SCP

Run this from Windows CMD, not inside the pod:

```bat
scp -P 11528 -i %USERPROFILE%\.ssh\id_ed25519 C:\D\4th\Thesis\data_small\data_small.tar.gz root@154.54.102.51:/tmp/
```

After upload, verify inside the pod:

```bash
cd /tmp
ls -lh data_small.tar.gz
```

Inspect archive structure before extraction:

```bash
tar -tzf data_small.tar.gz | head
```

Extract:

```bash
mkdir -p /tmp/data_small
tar -xzf /tmp/data_small.tar.gz -C /tmp/data_small
```

Verify:

```bash
tree -L 2 /tmp/data_small
find /tmp/data_small -type f | wc -l
```

### Upload Full Dataset Chunks

Large datasets may be uploaded in `.tar.gz` chunks.

Example:

```bat
scp -P 11528 -i %USERPROFILE%\.ssh\id_ed25519 C:\D\4th\Thesis\data_tars\us_xray\1.tar.gz root@154.54.102.51:/tmp/
```

Extract to a temporary folder first:

```bash
mkdir -p /tmp/data/us_xray/temp
tar -xzf /tmp/1.tar.gz -C /tmp/data/us_xray/temp
tree -L 2 --filelimit 50 /tmp/data/us_xray/temp
```

Then copy into both `/tmp` and `/workspace` dataset folders:

```bash
rsync -a --info=progress2 /tmp/data/us_xray/temp/ /tmp/data/us_xray/
rsync -a --info=progress2 /tmp/data/us_xray/temp/ /workspace/data/us_xray/
```

Clean temp contents but keep the folder:

```bash
find /tmp/data/us_xray/temp -mindepth 1 -delete
```

---

## Folder Structure

Recommended persistent structure:

```text
/workspace/
  MedCoSS/
  data/
  data_small/
  output_dir/
  checkpoints/
  logs/
```

Recommended active runtime structure:

```text
/tmp/
  data/
  data_small/
  output_dir/
  logs/
  snapshots/
  checkpoints/
```

---

## Dataset Structure

### Pretraining Dataset

Expected full dataset path:

```text
/tmp/data/
```

Expected small dataset path:

```text
/tmp/data_small/
```

Pretraining structure:

```text
us_report/
  master.csv

us_xray/
  *.png
  optional metadata/list JSON

us_pathology/
  *.tif
  pretrain_data_list.json
```

### Downstream Dataset

```text
ds_report/
  train.txt
  dev.txt
  test.txt

ds_xray/
  train/
    covid/
    normal/
    pneumonia/
  validation/
    covid/
    normal/
    pneumonia/

ds_pathology_cls/
  ADI/
  BACK/
  DEB/
  LYM/
  MUC/
  MUS/
  NORM/
  STR/
  TUM/

ds_pathology_seg/
  train/images
  train/labels
  test/images
  test/labels
```

---

## Full Dataset Counts Used

Prepared full dataset under `/tmp/data`:

### Upstream SSL

| Dataset | Role | Count |
|---|---|---:|
| `us_report/master.csv` | upstream report SSL | 199,897 usable records loaded |
| `us_xray` | upstream X-ray SSL | 112,120 `.png` images + 1 metadata/list file |
| `us_pathology` | upstream pathology SSL | 100,000 `.tif` images + `pretrain_data_list.json` |

### Downstream

| Dataset | Role | Count |
|---|---|---:|
| `ds_report` | PubMed20k sentence classification | train 180,040 / dev 30,212 / test 30,135 |
| `ds_xray` | X-ray classification | train 12,121 / validation 3,032 |
| `ds_pathology_cls` | CRC pathology classification | 7,180 images across 9 classes |
| `ds_pathology_seg` | GlaS segmentation | train 85 images + 85 labels, test 80 images + 80 labels |

Important:

- `ds_report` counts are sentence-level samples, not abstract counts.
- PubMed20k may be described as 15,000 training abstracts, but the MedCoSS downstream loader trains on labeled sentences. This is why logs show 180,040 training samples.

---

## Full Dataset Class Distributions

### `ds_xray`

```text
train:
  covid: 2892
  normal: 8153
  pneumonia: 1076
  total: 12121

validation:
  covid: 724
  normal: 2039
  pneumonia: 269
  total: 3032
```

### `ds_pathology_cls`

```text
ADI: 1338
BACK: 847
DEB: 339
LYM: 634
MUC: 1035
MUS: 592
NORM: 741
STR: 421
TUM: 1233
total: 7180
```

### `ds_pathology_seg`

```text
train/images: 85
train/labels: 85
test/images: 80
test/labels: 80
total files: 330
```

---

## Small Dataset Counts

Prepared small dataset under `/tmp/data_small` or local `C:\D\4th\Thesis\data_small`.

### Upstream SSL

| Dataset | Role | Count |
|---|---|---:|
| `us_report/master.csv` | upstream report SSL | 400 records |
| `us_xray` | upstream X-ray SSL | 400 `.png` images + 1 JSON |
| `us_pathology` | upstream pathology SSL | 400 `.tif` images + 1 JSON |

### Downstream

| Dataset | Role | Count |
|---|---|---:|
| `ds_report` | report classification | train 50 / dev 20 / test 20 |
| `ds_xray` | X-ray classification | train 150 / validation 60 |
| `ds_pathology_cls` | pathology classification | 450 images, 50 per class |
| `ds_pathology_seg` | GlaS-style segmentation | train 20 images + 20 labels, test 20 images + 20 labels |

---

## `pretrain_data_list.json` Handling

Do **not blindly delete** all `pretrain_data_list.json` files.

Some `pretrain_data_list.json` files are useful and expected, especially for large image folders where the dataloader uses the JSON list to know which files to load.

Check files:

```bash
find /tmp/data -name "pretrain_data_list.json" -ls
find /tmp/data_small -name "pretrain_data_list.json" -ls
```

Inspect contents:

```bash
head -20 /tmp/data/us_pathology/pretrain_data_list.json
```

Keep the file if it contains valid RunPod/Linux paths or valid relative image names.

Delete or regenerate only if it contains old local Windows paths such as:

```text
C:\D\4th\Thesis\...
```

or paths that do not exist on RunPod.

Example valid file:

```text
/tmp/data/us_pathology/pretrain_data_list.json
```

For the full upstream pathology dataset, this file is useful because the folder contains 100,000 `.tif` images.

---

## System Dependencies

Install system-level dependencies with `apt`, not `pip`.

```bash
apt update
apt install -y libgl1 libglib2.0-0 git curl openssh-server python3-pip rsync tree tmux htop net-tools
```

These are needed for:

- Git repository operations
- OpenCV runtime libraries
- SSH/SCP access
- Python package installation
- File copying and monitoring
- Long-running training sessions
- CPU/RAM/process monitoring with `htop`

---

## Python Version Issue

The selected container may come with:

```bash
Python 3.6.9
```

Python 3.6 is too old for several required packages, including:

- `tensorboard==2.14.0`
- `pandas==2.0.3`
- `matplotlib==3.7.5`
- `monai==1.3.0`

Therefore, install Python 3.8 and use a virtual environment.

---

## Install Python 3.8

Install helper tools:

```bash
apt update
apt install -y software-properties-common curl
```

Add the Deadsnakes PPA:

```bash
add-apt-repository ppa:deadsnakes/ppa -y
apt update
```

Install Python 3.8:

```bash
apt install -y python3.8 python3.8-dev python3.8-venv python3.8-distutils
```

Verify:

```bash
python3.8 --version
```

Expected output:

```text
Python 3.8.x
```

---

## Create Virtual Environment

Go to the MedCoSS repository:

```bash
cd /workspace/MedCoSS
```

Create virtual environment:

```bash
python3.8 -m venv venv
```

Activate it:

```bash
source venv/bin/activate
```

After activation, the terminal should show something like:

```text
(venv) root@...
```

Important:

- After activating the venv, use `python`, not system `python3`.
- `python` should point to the venv Python 3.8.

Check:

```bash
which python
python --version
```

---

## Install Pip in Virtual Environment

Install pip inside the venv:

```bash
curl -sS https://bootstrap.pypa.io/pip/3.8/get-pip.py | python
```

Upgrade pip tools:

```bash
python -m pip install --upgrade pip setuptools wheel
```

---

## Python Requirements

Keep only Python packages in `requirements_runpod.txt`.

Do **not** put `apt`, `add-apt-repository`, or shell setup commands inside `requirements_runpod.txt`.

Recommended `requirements_runpod.txt`:

```txt
tensorboard==2.14.0
tensorboard-data-server==0.7.2
timm==0.4.12
pandas==2.0.3
nltk==3.9.1
transformers==4.30.2
opencv-python==4.13.0.92

pydicom==2.4.4
matplotlib==3.7.5
nibabel==5.2.1
batchgenerators==0.25.1
einops==0.8.1
SimpleITK==2.2.1
tensorboardX==2.6.2.2
monai==1.3.0
medpy==0.5.2
```

Install requirements:

```bash
cd /workspace/MedCoSS
source venv/bin/activate

python -m pip install -r requirements_runpod.txt
```

Check OpenCV version:

```bash
python -c "import cv2; print(cv2.__version__)"
python -m pip show opencv-python
```

Expected package version:

```text
opencv-python==4.13.0.92
```

---

## Setup Script

Recommended `setup_runpod.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

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

echo "== Moving to MedCoSS repo =="
cd /workspace/MedCoSS

echo "== Activating venv =="
source venv/bin/activate

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
```

Run:

```bash
cd /workspace/MedCoSS
source venv/bin/activate
bash setup_runpod.sh
```

---

## NLTK Setup

Download required NLTK data:

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

---

## Installation Test

Run:

```bash
python -c "import torch, timm, pandas, cv2, transformers, monai; print('OK')"
```

If this prints:

```text
OK
```

then the main dependencies are installed correctly.

Also verify CUDA:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY
```

---

## First Run Validation

For full dataset:

```bash
find /tmp/data/us_xray -type f | wc -l
find /tmp/data/us_xray -type f -name "*.png" | wc -l

find /tmp/data/us_pathology -type f | wc -l
find /tmp/data/us_pathology -type f -name "*.tif" | wc -l

ls -lh /tmp/data/us_report/master.csv
tree -L 2 --filelimit 50 /tmp/data
```

For small dataset:

```bash
tree -L 2 --filelimit 50 /tmp/data_small
find /tmp/data_small -type f | wc -l
```

Check for empty files:

```bash
find /tmp/data -type f -empty
find /workspace/data -type f -empty
```

Check corrupted TIFFs if needed:

```bash
python - <<'PY'
from PIL import Image
from pathlib import Path

bad = []
for f in Path("/tmp/data/us_pathology").glob("*.tif"):
    try:
        with Image.open(f) as img:
            img.verify()
    except Exception:
        bad.append(str(f))

print("bad files:", len(bad))
for x in bad[:50]:
    print(x)
PY
```

---

## Stage-wise SSL Run

Use the stage-wise SSL script for safer recovery:

```bash
cd /workspace/MedCoSS
source venv/bin/activate
```

Run each stage:

```bash
bash run_ssl_stage.sh 1
bash run_ssl_stage.sh b1
bash run_ssl_stage.sh 2
bash run_ssl_stage.sh b2
bash run_ssl_stage.sh 3
```

Aliases are also supported:

```bash
bash run_ssl_stage.sh report
bash run_ssl_stage.sh report_buffer
bash run_ssl_stage.sh xray
bash run_ssl_stage.sh xray_buffer
bash run_ssl_stage.sh pathology
```

For long runs, use `tmux`:

```bash
tmux new -s ssl
bash run_ssl_stage.sh 1
```

Detach safely:

```text
Ctrl+b then d
```

Reattach:

```bash
tmux attach -t ssl
```

Check final SSL checkpoint:

```bash
ls -lh /tmp/output_dir/MedCoSS_Report_Xray_Path_buff_0.05_cen_0.01_2D_Path_300epoch/checkpoint-299.pth
```

Verify checkpoint load:

```bash
python - <<'PY'
import torch

ckpt = "/tmp/output_dir/MedCoSS_Report_Xray_Path_buff_0.05_cen_0.01_2D_Path_300epoch/checkpoint-299.pth"
obj = torch.load(ckpt, map_location="cpu")
print("OK:", ckpt)
print(obj.keys() if isinstance(obj, dict) else type(obj))
PY
```

---

## Stage-wise Downstream Run

After SSL Stage 3 completes, run downstream tasks stage-wise:

```bash
cd /workspace/MedCoSS
source venv/bin/activate
```

Run tasks:

```bash
bash run_ds_stage.sh 1          # PubMed20k report classification
bash run_ds_stage.sh 2          # Chest_XR classification
bash run_ds_stage.sh 3          # NCT_CRC_HE pathology classification
bash run_ds_stage.sh 4          # GlaS segmentation + evaluation
bash run_ds_stage.sh collect    # collect metrics
```

Aliases may include:

```bash
bash run_ds_stage.sh report
bash run_ds_stage.sh xray
bash run_ds_stage.sh pathcls
bash run_ds_stage.sh glas
bash run_ds_stage.sh metrics
```

For full runs, use three seeds:

```bash
DS_SEEDS=(0 10 100)
```

For debug runs, use one seed:

```bash
DS_SEEDS=(0)
```

---

## Metric Collection

Metric files are created under:

```text
/tmp/snapshots/**/metrics.json
```

Collect metrics into the run JSON:

```bash
python collect_runpod_metrics.py --run-file runpod_run_full.json --snapshot-root /tmp/snapshots
```

The updated collector should append results under the next numeric key:

```json
"0": { ... },
"1": { ... },
"2": { ... }
```

This preserves old runs instead of overwriting them.

---

## Monitoring

Monitor GPU usage:

```bash
watch -n 2 nvidia-smi
```

Monitor CPU/RAM/processes interactively:

```bash
htop
```

Useful interpretation:

- High GPU usage + active logs = healthy training.
- GPU low but CPU high = likely dataloader or KMeans CPU work.
- GPU low + CPU low + no logs = possibly stuck.
- During buffer KMeans, GPU may be 0% while CPU is high; this can be normal.

For a specific process:

```bash
ps aux | grep main_buffer_kmean
top -p <PID>     # example: top -p 203372
```

Check active Python/training processes:

```bash
ps aux | grep -E "main_pretrain|main_buffer|Downstream|torch.distributed|torchrun"
```

Monitor logs:

```bash
tail -f /tmp/logs/<stage>/log.txt
```

Find checkpoints:

```bash
find /tmp/output_dir -name "checkpoint-*.pth" -ls
```

Find metric files:

```bash
find /tmp/snapshots -name "metrics.json" -ls
```

Check disk usage:

```bash
df -h
du -sh /tmp/*
du -sh /workspace/*
```

---

## Expected Behavior

During SSL training:

- Logs should print periodically.
- Checkpoints should be saved.
- Final full-run checkpoint should look like:

```text
checkpoint-299.pth
```

During buffer creation:

- GPU may be used for feature extraction.
- KMeans runs mostly on CPU.
- GPU may show 0% during KMeans.
- Buffer files should be created:

```text
1D_text_0.01_0.05_kmean.csv
2D_xray_0.01_0.05_kmean.json
```

During downstream training:

- Each task writes outputs under `/tmp/snapshots`.
- Classification tasks write `metrics.json`.
- GlaS evaluation writes `metrics.json` under the prediction folder.

---

## Copy Outputs from `/tmp` to Persistent Storage

Outputs may be written to local container disk first:

```bash
OUTPUT_ROOT=/tmp/output_dir
LOG_ROOT=/tmp/logs
SNAPSHOT_ROOT=/tmp/snapshots
```

This helps avoid slow or unstable writes directly to mounted `/workspace`.

After the SSL/downstream run completes successfully, copy outputs back to persistent storage:

```bash
mkdir -p /workspace/output_dir /workspace/logs /workspace/MedCoSS/snapshots

rsync -ah --info=progress2 /tmp/output_dir/ /workspace/output_dir/
rsync -ah --info=progress2 /tmp/logs/ /workspace/logs/
rsync -ah --info=progress2 /tmp/snapshots/ /workspace/MedCoSS/snapshots/
```

Verify copied checkpoints:

```bash
find /workspace/output_dir -name "checkpoint-*.pth" -ls
```

Verify copied metrics:

```bash
find /workspace/MedCoSS/snapshots -name "metrics.json" -ls
```

If a copy is interrupted, rerun the same `rsync` command. It will continue/skip already copied files.

Once outputs are safely copied to `/workspace`, the `/tmp` copies may be removed if needed:

```bash
rm -rf /tmp/output_dir/*
rm -rf /tmp/logs/*
rm -rf /tmp/snapshots/*
```

---

## Cleanup Before Fresh Full Run

Stop old processes:

```bash
pkill -f main_pretrain || true
pkill -f main_buffer_kmean || true
pkill -f Downstream || true
pkill -f torch.distributed || true
pkill -f torchrun || true
```

Remove old debug SSL outputs:

```bash
rm -rf /tmp/output_dir/*
rm -rf /tmp/logs/*
```

Remove old downstream snapshots:

```bash
rm -rf /tmp/snapshots/*
```

Optional if copied debug outputs to workspace:

```bash
rm -rf /workspace/output_dir/*
rm -rf /workspace/logs/*
rm -rf /workspace/MedCoSS/snapshots/downstream/*
```

Clean old torchelastic temp folders:

```bash
rm -rf /tmp/torchelastic_*
```

Do **not** delete:

```text
/tmp/data
/workspace/data
/tmp/checkpoints
/workspace/checkpoints
/workspace/MedCoSS
```

---

## Workflow Strategy

### Phase 1 — Small Dataset

Use this first:

```text
/tmp/data_small/
```

Goal:

- Run the full pipeline.
- Catch dataloader/path/dependency errors.
- Confirm pretraining, downstream fine-tuning, and evaluation work end-to-end.

### Phase 2 — Full Dataset

Use this after the small dataset pipeline works:

```text
/tmp/data/
```

Goal:

- Run full SSL training.
- Save checkpoints and logs.
- Run downstream evaluation.
- Collect metrics.

---

## Pod Usage Rules

When idle:

- **STOP** the pod.

Do not **TERMINATE** the pod until:

- Checkpoints are saved.
- Logs are saved.
- Metrics are saved.
- Code is committed.
- Important outputs are copied to `/workspace`.

Important:

- Stopping the pod preserves `/workspace`.
- Terminating the pod deletes `/workspace`.

---

## Git Workflow

```bash
cd /workspace/MedCoSS

git status
git log --oneline -5

git add .
git commit -m "clean run setup"
```

---

## Common Issues

### `python: command not found`

Use:

```bash
python3 --version
```

or, after activating the venv:

```bash
source /workspace/MedCoSS/venv/bin/activate
python --version
```

### `No matching distribution found for tensorboard==2.14.0`

Cause:

- System Python is too old, usually Python 3.6.

Fix:

- Install Python 3.8.
- Create and activate a venv.
- Reinstall requirements inside the venv.

### `ssh: connect to host ... Connection refused`

Cause:

- SSH server is not running inside the pod.
- `openssh-server` may not be installed.

Fix:

```bash
apt update
apt install -y openssh-server
mkdir -p /run/sshd
ssh-keygen -A
/usr/sbin/sshd
```

### SSH asks for password

Cause:

- Public key is not added to `/root/.ssh/authorized_keys`.

Fix:

```bash
mkdir -p /root/.ssh
chmod 700 /root/.ssh

echo "PASTE_FULL_PUBLIC_KEY_HERE" >> /root/.ssh/authorized_keys

chmod 600 /root/.ssh/authorized_keys
pkill sshd || true
/usr/sbin/sshd
```

### `ss: command not found`

This is only a missing network utility. It does not mean SSH failed.

Install net tools if needed:

```bash
apt install -y net-tools
netstat -tlnp | grep :22
```

Or simply test SSH from Windows.

### SSH disconnect kills training

Cause:

- Command was run directly in SSH, not inside `tmux`.

Fix:

```bash
tmux new -s ssl
bash run_ssl_stage.sh 1
```

Detach:

```text
Ctrl+b then d
```

Reattach:

```bash
tmux attach -t ssl
```

### Buffer step appears stuck after feature extraction

If logs show something like:

```text
(199897, 768)
center number: ...
```

then the script is likely running KMeans on CPU.

Check CPU usage:

```bash
ps aux | grep main_buffer_kmean
ps -p <PID> -o pid,stat,etime,%cpu,%mem,cmd
```

GPU may show 0% during KMeans. This can be normal.

### Pathology image read error

Example:

```text
RuntimeError: Failed to read pathology image ... cannot identify image file
```

Check for empty files:

```bash
find /tmp/data/us_pathology -type f -empty
find /workspace/data/us_pathology -type f -empty
```

Replace corrupted/empty files from local source and verify:

```bash
file /tmp/FILE_NAME.tif
```

Then copy to both locations:

```bash
cp /tmp/FILE_NAME.tif /tmp/data/us_pathology/
cp /tmp/FILE_NAME.tif /workspace/data/us_pathology/
```

---

## Useful Commands

Check current directory:

```bash
pwd
```

Check disk usage:

```bash
df -h
```

Check folder size:

```bash
du -sh /tmp/data/*
du -sh /workspace/*
```

Check files:

```bash
ls -lh
```

Tree two levels deep:

```bash
tree -L 2
tree -L 2 /tmp/data
tree -L 2 /tmp/output_dir
```

Find files:

```bash
find /workspace -name "requirements_runpod.txt"
find /tmp/output_dir -name "checkpoint-*.pth"
find /tmp/snapshots -name "metrics.json"
```

Check Python path:

```bash
which python
python --version
```

Check pip path:

```bash
which pip
python -m pip --version
```

Check OpenCV version:

```bash
python -c "import cv2; print(cv2.__version__)"
python -m pip show opencv-python
```

Check GPU:

```bash
nvidia-smi
watch -n 2 nvidia-smi
```

Check running training processes:

```bash
ps aux | grep -E "main_pretrain|main_buffer|Downstream|torch.distributed|torchrun"
```

Kill old training processes if needed:

```bash
pkill -f main_pretrain || true
pkill -f main_buffer_kmean || true
pkill -f Downstream || true
pkill -f torch.distributed || true
pkill -f torchrun || true
```