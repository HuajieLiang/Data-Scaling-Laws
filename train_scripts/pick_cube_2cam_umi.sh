#!/usr/bin/env bash
# 使用 USB 腕部视角 + Top 俯视视角训练 pick_cube。
#
# 默认直接运行：
#   ./train_scripts/pick_cube_2cam_umi.sh
#
# 如需使用其他 Python 环境：
#   PYTHON_BIN=/path/to/python ./train_scripts/pick_cube_2cam_umi.sh
set -euo pipefail

# 无论从哪个工作目录调用脚本，都切换到 Data-Scaling-Laws 根目录，保证 Hydra
# 配置和 data/dataset/pick_cube_2cam 使用稳定的项目内相对路径。
project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
default_python_bin=/gpfs/huajieliang/conda_env/smolvla/bin/python
if [[ ! -x "$default_python_bin" ]]; then
  default_python_bin=/home/smore/miniconda3/envs/smolvla/bin/python
fi
python_bin=${PYTHON_BIN:-$default_python_bin}
dataset_path=${DATASET_PATH:-data/dataset_umi_zarr/pick_cube2_real_2cam/dataset.zarr.zip}

logging_time=$(date "+%d-%H.%M.%S")
now_date=$(date "+%Y.%m.%d")
run_dir="data/outputs/${now_date}/${logging_time: -8}_pick_cube_umi_2cam_align"

cd "$project_dir"

# 在启动 accelerate 前先检查数据，避免路径写错后等待到 Dataset 初始化才失败。
if [[ ! -f "$dataset_path" ]]; then
  echo "Dataset not found: $project_dir/$dataset_path" >&2
  exit 1
fi

echo "Dataset: $project_dir/$dataset_path"
echo "Output directory: $project_dir/$run_dir"

exec "$python_bin" -m accelerate.commands.launch --mixed_precision bf16 train.py \
  --config-name=train_diffusion_unet_timm_umi_workspace \
  "task=umi_2cam" \
  "multi_run.run_dir=$run_dir" \
  "multi_run.wandb_name_base=$logging_time" \
  "hydra.run.dir=$run_dir" \
  "hydra.sweep.dir=$run_dir" \
  "task.dataset_path=$dataset_path" \
  "task.dataset.dataset_idx=null" \
  "task.dataset.val_ratio=0.2" \
  "training.num_epochs=200" \
  "training.gradient_accumulate_every=4" \
  "dataloader.batch_size=256" \
  "val_dataloader.batch_size=128" \
  "logging.mode=offline" \
  "logging.name=${logging_time}_pick_cube_umi_2cam_align" \
  "policy.obs_encoder.model_name=vit_base_patch14_dinov2.lvd142m" \
  "policy.obs_encoder.share_rgb_model=true" \
  "checkpoint.topk.k=100" \
  "task.obs_down_sample_steps=3" \
  "task.action_down_sample_steps=3" \
  "task.dataset_frequeny=23.129633199732353"
