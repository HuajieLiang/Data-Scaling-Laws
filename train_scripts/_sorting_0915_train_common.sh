#!/usr/bin/env bash
# Shared launcher for sorting_0915 bimanual UMI training variants.
set -euo pipefail

: "${SORTING_DATASET_PATH:?SORTING_DATASET_PATH is required}"
: "${SORTING_CAMERA_COUNT:?SORTING_CAMERA_COUNT must be 2 or 3}"
: "${SORTING_RUN_TAG:?SORTING_RUN_TAG is required}"
: "${SORTING_LEROBOT_SOURCE_PATH:?SORTING_LEROBOT_SOURCE_PATH is required}"
: "${SORTING_DATASET_FREQUENCY:?SORTING_DATASET_FREQUENCY is required}"
: "${SORTING_OBS_DOWNSAMPLE:?SORTING_OBS_DOWNSAMPLE is required}"
: "${SORTING_ACTION_DOWNSAMPLE:?SORTING_ACTION_DOWNSAMPLE is required}"

project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
default_python_bin=/gpfs/huajieliang/conda_env/smolvla/bin/python
if [[ ! -x "$default_python_bin" ]]; then
  default_python_bin=/home/smore/miniconda3/envs/smolvla/bin/python
fi
python_bin=${PYTHON_BIN:-$default_python_bin}
mixed_precision=${MIXED_PRECISION:-bf16}

display_path() {
  local path=$1
  if [[ "$path" = /* ]]; then
    printf "%s\n" "$path"
  else
    printf "%s/%s\n" "$project_dir" "$path"
  fi
}

case "$SORTING_CAMERA_COUNT" in
  2|3) ;;
  *)
    echo "SORTING_CAMERA_COUNT must be 2 or 3 (got $SORTING_CAMERA_COUNT)" >&2
    exit 2
    ;;
esac
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
if [[ ! -d "$SORTING_LEROBOT_SOURCE_PATH" ]]; then
  echo "LeRobot source not found: $(display_path "$SORTING_LEROBOT_SOURCE_PATH")" >&2
  exit 1
fi
if [[ ! -f "$SORTING_DATASET_PATH" ]]; then
  echo "Zarr dataset not found: $(display_path "$SORTING_DATASET_PATH")" >&2
  exit 1
fi
"$python_bin" train_scripts/validate_sorting_0915_dataset.py \
  "$SORTING_DATASET_PATH" --camera-count "$SORTING_CAMERA_COUNT"

logging_time=$(date "+%d-%H.%M.%S")
now_date=$(date "+%Y.%m.%d")
run_dir="data/outputs/${now_date}/${logging_time}_${SORTING_RUN_TAG}"

echo "LeRobot source: $(display_path "$SORTING_LEROBOT_SOURCE_PATH")"
echo "Training Zarr: $(display_path "$SORTING_DATASET_PATH")"
echo "Task: sorting_0915 bimanual UMI (${SORTING_CAMERA_COUNT} cameras)"
echo "Pose convention: per-episode left-frame0 common frame; sampled as current-TCP-relative"
echo "Mixed precision: $mixed_precision"
echo "Output directory: $project_dir/$run_dir"

task_overrides=(
  "task.name=sorting_0915_bimanual_${SORTING_CAMERA_COUNT}cam"
  "task.dataset_path=$SORTING_DATASET_PATH"
  "task.dataset.dataset_path=$SORTING_DATASET_PATH"
  "task.dataset.shape_meta=\${task.shape_meta}"
  "task.dataset.pose_repr=\${task.pose_repr}"
  "task.dataset.val_ratio=${VAL_RATIO:-0.2}"
  "+task.dataset.use_ratio=${USE_RATIO:-1.0}"
  "+task.dataset.dataset_idx=${DATASET_IDX:-null}"
  "task.ignore_proprioception=${IGNORE_PROPRIOCEPTION:-false}"
  "task.dataset_frequeny=$SORTING_DATASET_FREQUENCY"
  "task.obs_down_sample_steps=$SORTING_OBS_DOWNSAMPLE"
  "+task.action_down_sample_steps=$SORTING_ACTION_DOWNSAMPLE"
  "task.shape_meta.action.down_sample_steps=\${task.action_down_sample_steps}"
  "task.camera_obs_latency=0"
  "task.robot_obs_latency=0"
  "task.gripper_obs_latency=0"
  "task.pose_repr.obs_pose_repr=relative"
  "task.pose_repr.action_pose_repr=relative"
)

if [[ "$SORTING_CAMERA_COUNT" == 3 ]]; then
  task_overrides+=(
    "+task.shape_meta.obs.camera2_rgb={shape:[3,224,224],horizon:\${task.img_obs_horizon},latency_steps:0,down_sample_steps:\${task.obs_down_sample_steps},type:rgb,ignore_by_policy:false}"
  )
fi

exec "$python_bin" -m accelerate.commands.launch \
  --mixed_precision "$mixed_precision" \
  train.py \
  --config-name=train_diffusion_unet_umi_bimanual_workspace \
  "multi_run.run_dir=$run_dir" \
  "multi_run.wandb_name_base=$logging_time" \
  "hydra.run.dir=$run_dir" \
  "hydra.sweep.dir=$run_dir" \
  "${task_overrides[@]}" \
  "training.num_epochs=${NUM_EPOCHS:-200}" \
  "training.checkpoint_every=${CHECKPOINT_EVERY:-10}" \
  "+training.layer_decay=${LAYER_DECAY:-1.0}" \
  "+training.encoder_lr_coefficient=${ENCODER_LR_COEFFICIENT:-0.1}" \
  "+training.use_in_the_wild_val=${USE_IN_THE_WILD_VAL:-false}" \
  "+training.wild_sample_every=${WILD_SAMPLE_EVERY:-5}" \
  "+training.in_the_wild_type=${IN_THE_WILD_TYPE:-[seen_object-unseen_env,unseen_object-seen_env,unseen_object-unseen_env]}" \
  "checkpoint.topk.k=${CHECKPOINT_TOPK_K:-100}" \
  "+checkpoint.only_save_recent=${ONLY_SAVE_RECENT:-false}" \
  "training.gradient_accumulate_every=${GRADIENT_ACCUMULATE_EVERY:-4}" \
  "dataloader.batch_size=${BATCH_SIZE:-16}" \
  "val_dataloader.batch_size=${VAL_BATCH_SIZE:-8}" \
  "logging.mode=${LOGGING_MODE:-offline}" \
  "logging.name=${logging_time}_${SORTING_RUN_TAG}" \
  "policy.obs_encoder.model_name=${OBS_ENCODER_MODEL:-vit_base_patch16_clip_224.openai}" \
  "+policy.obs_encoder.use_lora=${USE_LORA:-false}" \
  "policy.obs_encoder.feature_aggregation=${OBS_FEATURE_AGGREGATION:-cls_token}" \
  "policy.obs_encoder.share_rgb_model=${SHARE_RGB_MODEL:-true}" \
  "policy.obs_encoder.transforms=[{type:RandomCrop,ratio:0.95},{_target_:torchvision.transforms.ColorJitter,brightness:0.3,contrast:0.4,saturation:0.5,hue:0.08}]"
