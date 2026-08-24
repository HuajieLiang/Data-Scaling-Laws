#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${PYTHON_BIN:-/home/smore/miniconda3/envs/smolvla/bin/python}
dataset_path=${DATASET_PATH:-data/dataset_umi_zarr/pick_cube2_real_1cam/dataset.zarr.zip}
logging_time=$(date "+%d-%H.%M.%S")
now_date=$(date "+%Y.%m.%d")
run_dir="data/outputs/${now_date}/${logging_time: -8}_pick_cube_base_relative_1cam"

cd "$project_dir"
if [[ ! -f "$dataset_path" ]]; then
  echo "Dataset not found: $project_dir/$dataset_path" >&2
  exit 1
fi
echo "Dataset: $project_dir/$dataset_path"
echo "Output directory: $project_dir/$run_dir"

exec "$python_bin" -m accelerate.commands.launch --mixed_precision bf16 train.py \
  --config-name=train_diffusion_unet_timm_umi_workspace \
  "multi_run.run_dir=$run_dir" \
  "multi_run.wandb_name_base=$logging_time" \
  "hydra.run.dir=$run_dir" \
  "hydra.sweep.dir=$run_dir" \
  "task.dataset_path=$dataset_path" \
  "task.dataset.dataset_idx=null" \
  "task.dataset.val_ratio=0.2" \
  "training.num_epochs=120" \
  "training.gradient_accumulate_every=2" \
  "dataloader.batch_size=16" \
  "val_dataloader.batch_size=16" \
  "logging.mode=offline" \
  "logging.name=${logging_time}_pick_cube_base_relative_1cam" \
  "policy.obs_encoder.model_name=vit_base_patch14_dinov2.lvd142m"
