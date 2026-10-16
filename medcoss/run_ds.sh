#!/usr/bin/env bash
# =============================================================================
# MedCoSS downstream fine-tuning (RunPod) — 4 tasks only:
#   PubMed20k (report) | Chest_XR | NCT_CRC_HE (pathology cls) | GlaS (pathology seg)
# =============================================================================

set -euo pipefail

# Resolve legacy relative source paths from the baseline, regardless of caller CWD.
MEDCOSS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${MEDCOSS_ROOT}"

# --- Run length ---
EPOCHS=1
LAST_EPOCH=$((EPOCHS - 1))
RUN_TAG="${EPOCHS}epoch"
DS_REPORTS_EPOCHS=5
DS_XRAY_EPOCHS=80
DS_PATHOLOGYCLS_EPOCHS=10
DS_PATHOLOGYSEG_EPOCHS=100
DS_SEEDS=(0)

DATA_ROOT="${DATA_ROOT:-/tmp/data}"
DS_REPORT="${DS_REPORT:-${DATA_ROOT}/ds_report}"
DS_XRAY="${DS_XRAY:-${DATA_ROOT}/ds_xray}"
DS_PATH_CLS="${DS_PATH_CLS:-${DATA_ROOT}/ds_pathology_cls}"
DS_PATH_SEG="${DS_PATH_SEG:-${DATA_ROOT}/ds_pathology_seg}"
SNAPSHOT_ROOT="${SNAPSHOT_ROOT:-/tmp/snapshots}"

PRETRAINED_DIR="${PRETRAINED_DIR:-/tmp/output_dir/MedCoSS_Report_Xray_Path_buff_0.05_cen_0.01_2D_Path_${RUN_TAG}}"
PRETRAINED_CKPT="${PRETRAINED_CKPT:-${PRETRAINED_DIR}/checkpoint-${LAST_EPOCH}.pth}"

test -f "$PRETRAINED_CKPT" || { echo "Missing pretrained checkpoint: $PRETRAINED_CKPT"; exit 1; }

gpu_id=0

reload_from_pretrained=True
pretrained_path="${PRETRAINED_CKPT}"
exp_name='MedCoSS_Report_Xray_Path_buff_0.05'


########################################################################################################################################
# 1) PubMed20k — report classification (Dim_1)
########################################################################################################################################

task_id='1D_PubMed'

model_name='model'
data_path="${DS_REPORT}"

lr=0.0002


seed=0
meid='_'$exp_name'/seed_'$seed'/lr_'$lr'/'

path_id=$task_id$meid
echo $task_id" Training - shallow"
snapshot_dir="${SNAPSHOT_ROOT}/downstream/dim_1/${path_id}"
mkdir -p $snapshot_dir
CUDA_VISIBLE_DEVICES=$gpu_id python -u Downstream/Dim_1/PudMed20k/main.py \
--arch='unified_vit' \
--data_path=$data_path \
--snapshot_dir=$snapshot_dir \
--input_size=112 \
--batch_size=64 \
--num_gpus=1 \
--num_epochs=$DS_REPORTS_EPOCHS \
--start_epoch=0 \
--learning_rate=$lr \
--num_classes=5 \
--num_workers=32 \
--reload_from_pretrained=$reload_from_pretrained \
--pretrained_path=$pretrained_path \
--val_only=0 \
--random_seed=$seed \
--model_name=$model_name 


seed=10
meid='_'$exp_name'/seed_'$seed'/lr_'$lr'/'

path_id=$task_id$meid
echo $task_id" Training - shallow"
snapshot_dir="${SNAPSHOT_ROOT}/downstream/dim_1/${path_id}"
mkdir -p $snapshot_dir
CUDA_VISIBLE_DEVICES=$gpu_id python -u Downstream/Dim_1/PudMed20k/main.py \
--arch='unified_vit' \
--data_path=$data_path \
--snapshot_dir=$snapshot_dir \
--input_size=112 \
--batch_size=64 \
--num_gpus=1 \
--num_epochs=$DS_REPORTS_EPOCHS \
--start_epoch=0 \
--learning_rate=$lr \
--num_classes=5 \
--num_workers=32 \
--reload_from_pretrained=$reload_from_pretrained \
--pretrained_path=$pretrained_path \
--val_only=0 \
--random_seed=$seed \
--model_name=$model_name 


seed=100
meid='_'$exp_name'/seed_'$seed'/lr_'$lr'/'

path_id=$task_id$meid
echo $task_id" Training - shallow"
snapshot_dir="${SNAPSHOT_ROOT}/downstream/dim_1/${path_id}"
mkdir -p $snapshot_dir
CUDA_VISIBLE_DEVICES=$gpu_id python -u Downstream/Dim_1/PudMed20k/main.py \
--arch='unified_vit' \
--data_path=$data_path \
--snapshot_dir=$snapshot_dir \
--input_size=112 \
--batch_size=64 \
--num_gpus=1 \
--num_epochs=$DS_REPORTS_EPOCHS \
--start_epoch=0 \
--learning_rate=$lr \
--num_classes=5 \
--num_workers=32 \
--reload_from_pretrained=$reload_from_pretrained \
--pretrained_path=$pretrained_path \
--val_only=0 \
--random_seed=$seed \
--model_name=$model_name 


########################################################################################################################################
# 2) Chest_XR — X-ray classification (Dim_2)
########################################################################################################################################

task_id='2D_ChestXR'

data_path="${DS_XRAY}"
lr=0.00005


seed=0
meid='_'$exp_name'/seed_'$seed'/lr_'$lr'/'

path_id=$task_id$meid
echo $task_id" Training - shallow"
snapshot_dir="${SNAPSHOT_ROOT}/downstream/dim_2/${path_id}"
mkdir -p $snapshot_dir
CUDA_VISIBLE_DEVICES=$gpu_id python -u Downstream/Dim_2/Chest_XR/main.py \
--arch='unified_vit' \
--data_path=$data_path \
--snapshot_dir=$snapshot_dir \
--input_size='224,224' \
--batch_size=32 \
--num_gpus=1 \
--num_epochs=$DS_XRAY_EPOCHS \
--start_epoch=0 \
--learning_rate=$lr \
--num_classes=3 \
--num_workers=32 \
--reload_from_pretrained=$reload_from_pretrained \
--pretrained_path=$pretrained_path \
--val_only=0 \
--random_seed=$seed 

seed=10
meid='_'$exp_name'/seed_'$seed'/lr_'$lr'/'

path_id=$task_id$meid
echo $task_id" Training - shallow"
snapshot_dir="${SNAPSHOT_ROOT}/downstream/dim_2/${path_id}"
mkdir -p $snapshot_dir
CUDA_VISIBLE_DEVICES=$gpu_id python -u Downstream/Dim_2/Chest_XR/main.py \
--arch='unified_vit' \
--data_path=$data_path \
--snapshot_dir=$snapshot_dir \
--input_size='224,224' \
--batch_size=32 \
--num_gpus=1 \
--num_epochs=$DS_XRAY_EPOCHS \
--start_epoch=0 \
--learning_rate=$lr \
--num_classes=3 \
--num_workers=32 \
--reload_from_pretrained=$reload_from_pretrained \
--pretrained_path=$pretrained_path \
--val_only=0 \
--random_seed=$seed 


seed=100
meid='_'$exp_name'/seed_'$seed'/lr_'$lr'/'

path_id=$task_id$meid
echo $task_id" Training - shallow"
snapshot_dir="${SNAPSHOT_ROOT}/downstream/dim_2/${path_id}"
mkdir -p $snapshot_dir
CUDA_VISIBLE_DEVICES=$gpu_id python -u Downstream/Dim_2/Chest_XR/main.py \
--arch='unified_vit' \
--data_path=$data_path \
--snapshot_dir=$snapshot_dir \
--input_size='224,224' \
--batch_size=32 \
--num_gpus=1 \
--num_epochs=$DS_XRAY_EPOCHS \
--start_epoch=0 \
--learning_rate=$lr \
--num_classes=3 \
--num_workers=32 \
--reload_from_pretrained=$reload_from_pretrained \
--pretrained_path=$pretrained_path \
--val_only=0 \
--random_seed=$seed 


##############################################################################################################
# 3) NCT_CRC_HE — pathology classification (Dim_2)
##############################################################################################################

task_id='2D_NCT_CRC_HE'

data_path="${DS_PATH_CLS}"

lr=0.0001

seed=0
meid='_'$exp_name'/seed_'$seed'/lr_'$lr'/'

path_id=$task_id$meid
echo $task_id" Training - shallow"
snapshot_dir="${SNAPSHOT_ROOT}/downstream/dim_2/${path_id}"
mkdir -p $snapshot_dir
CUDA_VISIBLE_DEVICES=$gpu_id python -u Downstream/Dim_2/NCT_CRC_HE/main.py \
--arch='unified_vit' \
--data_path=$data_path \
--snapshot_dir=$snapshot_dir \
--input_size='224,224' \
--batch_size=32 \
--num_gpus=1 \
--num_epochs=$DS_PATHOLOGYCLS_EPOCHS \
--start_epoch=0 \
--learning_rate=$lr \
--num_classes=9 \
--num_workers=32 \
--reload_from_pretrained=$reload_from_pretrained \
--pretrained_path=$pretrained_path \
--val_only=0 \
--random_seed=$seed 

seed=10
meid='_'$exp_name'/seed_'$seed'/lr_'$lr'/'

path_id=$task_id$meid
echo $task_id" Training - shallow"
snapshot_dir="${SNAPSHOT_ROOT}/downstream/dim_2/${path_id}"
mkdir -p $snapshot_dir
CUDA_VISIBLE_DEVICES=$gpu_id python -u Downstream/Dim_2/NCT_CRC_HE/main.py \
--arch='unified_vit' \
--data_path=$data_path \
--snapshot_dir=$snapshot_dir \
--input_size='224,224' \
--batch_size=32 \
--num_gpus=1 \
--num_epochs=$DS_PATHOLOGYCLS_EPOCHS \
--start_epoch=0 \
--learning_rate=$lr \
--num_classes=9 \
--num_workers=32 \
--reload_from_pretrained=$reload_from_pretrained \
--pretrained_path=$pretrained_path \
--val_only=0 \
--random_seed=$seed 

seed=100
meid='_'$exp_name'/seed_'$seed'/lr_'$lr'/'

path_id=$task_id$meid
echo $task_id" Training - shallow"
snapshot_dir="${SNAPSHOT_ROOT}/downstream/dim_2/${path_id}"
mkdir -p $snapshot_dir
CUDA_VISIBLE_DEVICES=$gpu_id python -u Downstream/Dim_2/NCT_CRC_HE/main.py \
--arch='unified_vit' \
--data_path=$data_path \
--snapshot_dir=$snapshot_dir \
--input_size='224,224' \
--batch_size=32 \
--num_gpus=1 \
--num_epochs=$DS_PATHOLOGYCLS_EPOCHS \
--start_epoch=0 \
--learning_rate=$lr \
--num_classes=9 \
--num_workers=32 \
--reload_from_pretrained=$reload_from_pretrained \
--pretrained_path=$pretrained_path \
--val_only=0 \
--random_seed=$seed 

######################################################################################################
# 4) GlaS — pathology segmentation (Dim_2)
######################################################################################################

nnudata="${DS_PATH_SEG}"

gpu_id=1

task_id='Glas'

lr=0.0001


seed=0
meid='_'$exp_name'/seed_'$seed'/lr_'$lr'/'
path_id=$task_id$meid
echo $task_id" Training - shallow"
snapshot_dir="${SNAPSHOT_ROOT}/downstream/dim_2/${path_id}"
mkdir -p $snapshot_dir
CUDA_VISIBLE_DEVICES=$gpu_id python -u Downstream/Dim_2/Glas/train.py \
--arch='unified_vit' \
--data_dir=$nnudata \
--snapshot_dir=$snapshot_dir \
--nnUNet_preprocessed=$nnudata \
--input_size='512,512' \
--batch_size=4 \
--num_gpus=1 \
--num_epochs=$DS_PATHOLOGYSEG_EPOCHS \
--start_epoch=0 \
--learning_rate=$lr \
--num_classes=1 \
--num_workers=10 \
--weight_std=False \
--random_seed=$seed \
--reload_from_pretrained=$reload_from_pretrained \
--pretrained_path=$pretrained_path 

echo $task_id" Evaluating"
output_dir="${SNAPSHOT_ROOT}/downstream/dim_2/${path_id}/prediction/"
mkdir -p $output_dir
CUDA_VISIBLE_DEVICES=$gpu_id python -u Downstream/Dim_2/Glas/evaluate.py \
--arch='unified_vit' \
--data_dir=$nnudata \
--nnUNet_preprocessed=$nnudata \
--reload_from_checkpoint=True \
--checkpoint_path="${snapshot_dir}/checkpoint.pth" \
--save_path=$output_dir \
--input_size='512,512' \
--batch_size=1 \
--num_classes=1 \
--num_gpus=1 \
--FP16=False \
--random_seed=$seed \
--weight_std=False \
--isHD=True 


seed=10
meid='_'$exp_name'/seed_'$seed'/lr_'$lr'/'
path_id=$task_id$meid
echo $task_id" Training - shallow"
snapshot_dir="${SNAPSHOT_ROOT}/downstream/dim_2/${path_id}"
mkdir -p $snapshot_dir
CUDA_VISIBLE_DEVICES=$gpu_id python -u Downstream/Dim_2/Glas/train.py \
--arch='unified_vit' \
--data_dir=$nnudata \
--snapshot_dir=$snapshot_dir \
--nnUNet_preprocessed=$nnudata \
--input_size='512,512' \
--batch_size=4 \
--num_gpus=1 \
--num_epochs=$DS_PATHOLOGYSEG_EPOCHS \
--start_epoch=0 \
--learning_rate=$lr \
--num_classes=1 \
--num_workers=10 \
--weight_std=False \
--random_seed=$seed \
--reload_from_pretrained=$reload_from_pretrained \
--pretrained_path=$pretrained_path

echo $task_id" Evaluating"
output_dir="${SNAPSHOT_ROOT}/downstream/dim_2/${path_id}/prediction/"
mkdir -p $output_dir
CUDA_VISIBLE_DEVICES=$gpu_id python -u Downstream/Dim_2/Glas/evaluate.py \
--arch='unified_vit' \
--data_dir=$nnudata \
--nnUNet_preprocessed=$nnudata \
--reload_from_checkpoint=True \
--checkpoint_path="${snapshot_dir}/checkpoint.pth" \
--save_path=$output_dir \
--input_size='512,512' \
--batch_size=1 \
--num_classes=1 \
--num_gpus=1 \
--FP16=False \
--random_seed=$seed \
--weight_std=False \
--isHD=True 


seed=100
meid='_'$exp_name'/seed_'$seed'/lr_'$lr'/'
path_id=$task_id$meid
echo $task_id" Training - shallow"
snapshot_dir="${SNAPSHOT_ROOT}/downstream/dim_2/${path_id}"
mkdir -p $snapshot_dir
CUDA_VISIBLE_DEVICES=$gpu_id python -u Downstream/Dim_2/Glas/train.py \
--arch='unified_vit' \
--data_dir=$nnudata \
--snapshot_dir=$snapshot_dir \
--nnUNet_preprocessed=$nnudata \
--input_size='512,512' \
--batch_size=4 \
--num_gpus=1 \
--num_epochs=$DS_PATHOLOGYSEG_EPOCHS \
--start_epoch=0 \
--learning_rate=$lr \
--num_classes=1 \
--num_workers=10 \
--weight_std=False \
--random_seed=$seed \
--reload_from_pretrained=$reload_from_pretrained \
--pretrained_path=$pretrained_path

echo $task_id" Evaluating"
output_dir="${SNAPSHOT_ROOT}/downstream/dim_2/${path_id}/prediction/"
mkdir -p $output_dir
CUDA_VISIBLE_DEVICES=$gpu_id python -u Downstream/Dim_2/Glas/evaluate.py \
--arch='unified_vit' \
--data_dir=$nnudata \
--nnUNet_preprocessed=$nnudata \
--reload_from_checkpoint=True \
--checkpoint_path="${snapshot_dir}/checkpoint.pth" \
--save_path=$output_dir \
--input_size='512,512' \
--batch_size=1 \
--num_classes=1 \
--num_gpus=1 \
--FP16=False \
--random_seed=$seed \
--weight_std=False \
--isHD=True 

########################################################################################################################################
# Refresh run JSON from all metrics.json under snapshots/ (seed_results, averages, raw_metrics).
########################################################################################################################################
RUNPOD_RUN_JSON="${RUNPOD_RUN_JSON:-runpod_run_full.json}"
if command -v python >/dev/null 2>&1; then
  python collect_runpod_metrics.py --run-file "$RUNPOD_RUN_JSON" --snapshot-root "$SNAPSHOT_ROOT" \
    || echo "WARNING: collect_runpod_metrics.py failed (run JSON not updated)" >&2
fi

