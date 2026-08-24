#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${PYTHON_BIN:-/gpfs/huajieliang/conda_env/smolvla/bin/python}
logging_time=$(date "+%d-%H.%M.%S")
now_date=$(date "+%Y.%m.%d")
run_dir="data/outputs/${now_date}/${logging_time: -8}_pick_cube_umi_1cam_align"

cd "$project_dir"
echo "Output directory: $project_dir/$run_dir"

# CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

exec "$python_bin" -m accelerate.commands.launch --mixed_precision bf16 train.py \
  --config-name=train_diffusion_unet_timm_umi_workspace \
  "multi_run.run_dir=$run_dir" \
  "multi_run.wandb_name_base=$logging_time" \
  "hydra.run.dir=$run_dir" \
  "hydra.sweep.dir=$run_dir" \
  "task.dataset_path=data/dataset_umi_zarr/pick_cube2_real_1cam/dataset.zarr.zip" \
  "task.dataset.dataset_idx=null" \
  "task.dataset.val_ratio=0.2" \
  "training.num_epochs=200" \
  "training.gradient_accumulate_every=2" \
  "dataloader.batch_size=512" \
  "val_dataloader.batch_size=128" \
  "logging.mode=offline" \
  "logging.name=${logging_time}_pick_cube_umi_1cam_align" \
  "policy.obs_encoder.model_name=vit_base_patch14_dinov2.lvd142m" \
  "checkpoint.topk.k=100" \
  "task.obs_down_sample_steps=3" \
  "task.action_down_sample_steps=3" \
  "task.dataset_frequeny=23.129633199732353" 
