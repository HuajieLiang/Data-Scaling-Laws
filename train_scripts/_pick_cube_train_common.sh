#!/usr/bin/env bash
# Shared launcher for all pick_cube training variants.
#
# The public wrappers in this directory only set the dataset/source, camera
# count and whether proprioceptive TCP coordinates are fed to the policy.
# Keeping the actual accelerate/Hydra invocation here prevents the variants
# from silently drifting apart.
set -euo pipefail

: "${PICK_CUBE_DATASET_PATH:?PICK_CUBE_DATASET_PATH is required}"
: "${PICK_CUBE_TASK:?PICK_CUBE_TASK must be umi or umi_2cam}"
: "${PICK_CUBE_RUN_TAG:?PICK_CUBE_RUN_TAG is required}"
: "${PICK_CUBE_SOURCE:?PICK_CUBE_SOURCE must be umi or real}"
: "${PICK_CUBE_IGNORE_PROPRIOCEPTION:?PICK_CUBE_IGNORE_PROPRIOCEPTION must be true or false}"
: "${PICK_CUBE_DATASET_FREQUENCY:?PICK_CUBE_DATASET_FREQUENCY is required}"
: "${PICK_CUBE_OBS_DOWNSAMPLE:?PICK_CUBE_OBS_DOWNSAMPLE is required}"
: "${PICK_CUBE_ACTION_DOWNSAMPLE:?PICK_CUBE_ACTION_DOWNSAMPLE is required}"

project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
default_python_bin=/gpfs/huajieliang/conda_env/smolvla/bin/python
if [[ ! -x "$default_python_bin" ]]; then
  default_python_bin=/home/smore/miniconda3/envs/smolvla/bin/python
fi
python_bin=${PYTHON_BIN:-$default_python_bin}
mixed_precision=${MIXED_PRECISION:-bf16}

case "$mixed_precision" in
  bf16|fp16|no) ;;
  *)
    echo "MIXED_PRECISION must be one of: bf16, fp16, no (got $mixed_precision)" >&2
    exit 2
    ;;
esac
if [[ ! -x "$python_bin" ]]; then
  echo "Python executable not found or not executable: $python_bin" >&2
  exit 1
fi

cd "$project_dir"
if [[ ! -f "$PICK_CUBE_DATASET_PATH" ]]; then
  echo "Dataset not found: $project_dir/$PICK_CUBE_DATASET_PATH" >&2
  exit 1
fi
camera_count=1
if [[ "$PICK_CUBE_TASK" == umi_2cam ]]; then
  camera_count=2
fi
"$python_bin" train_scripts/validate_pick_cube_dataset.py \
  "$PICK_CUBE_DATASET_PATH" --camera-count "$camera_count"

logging_time=$(date "+%d-%H.%M.%S")
now_date=$(date "+%Y.%m.%d")
run_dir="data/outputs/${now_date}/${logging_time}_${PICK_CUBE_RUN_TAG}"

echo "Source: $PICK_CUBE_SOURCE"
echo "Dataset: $project_dir/$PICK_CUBE_DATASET_PATH"
echo "Task: $PICK_CUBE_TASK"
echo "Disk pose: absolute TCP in fixed table axes (+X forward, +Y left, +Z up)"
echo "Model pose: current-TCP-relative observation and action"
echo "Proprioception: $([[ "$PICK_CUBE_IGNORE_PROPRIOCEPTION" == true ]] && echo disabled || echo enabled)"
echo "Mixed precision: $mixed_precision"
echo "Output directory: $project_dir/$run_dir"

task_overrides=(
  "task=$PICK_CUBE_TASK"
  "task.dataset_path=$PICK_CUBE_DATASET_PATH"
  "task.dataset.dataset_idx=null"
  "task.dataset.val_ratio=0.2"
  "task.dataset.use_ratio=1.0"
  "task.ignore_proprioception=$PICK_CUBE_IGNORE_PROPRIOCEPTION"
  "task.dataset_frequeny=$PICK_CUBE_DATASET_FREQUENCY"
  "task.obs_down_sample_steps=$PICK_CUBE_OBS_DOWNSAMPLE"
  "task.action_down_sample_steps=$PICK_CUBE_ACTION_DOWNSAMPLE"
  "task.pose_repr.obs_pose_repr=relative"
  "task.pose_repr.action_pose_repr=relative"
)

if [[ "$PICK_CUBE_TASK" == umi_2cam ]]; then
  task_overrides+=("policy.obs_encoder.share_rgb_model=true")
fi
exec "$python_bin" -m accelerate.commands.launch \
  --mixed_precision "$mixed_precision" \
  train.py \
  --config-name=train_diffusion_unet_timm_umi_workspace \
  "multi_run.run_dir=$run_dir" \
  "multi_run.wandb_name_base=$logging_time" \
  "hydra.run.dir=$run_dir" \
  "hydra.sweep.dir=$run_dir" \
  "${task_overrides[@]}" \
  "training.num_epochs=${NUM_EPOCHS:-200}" \
  "training.checkpoint_every=${CHECKPOINT_EVERY:-10}" \
  "checkpoint.topk.k=${CHECKPOINT_TOPK_K:-100}" \
  "training.gradient_accumulate_every=${GRADIENT_ACCUMULATE_EVERY:-2}" \
  "dataloader.batch_size=${BATCH_SIZE:-512}" \
  "val_dataloader.batch_size=${VAL_BATCH_SIZE:-128}" \
  "logging.mode=${LOGGING_MODE:-offline}" \
  "logging.name=${logging_time}_${PICK_CUBE_RUN_TAG}" \
  "policy.obs_encoder.model_name=${OBS_ENCODER_MODEL:-vit_base_patch14_dinov2.lvd142m}"
