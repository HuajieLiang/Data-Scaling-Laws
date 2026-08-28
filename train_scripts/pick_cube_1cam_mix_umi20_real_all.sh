#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PICK_CUBE_SOURCE=mix_umi20_real_all
export PICK_CUBE_TASK=umi
export PICK_CUBE_DATASET_PATH=${DATASET_PATH:-data/dataset_mix_zarr/pick_cube_1cam_umi20_real_all/dataset.zarr.zip}
export PICK_CUBE_IGNORE_PROPRIOCEPTION=false
# One scalar is required by the current UmiDataset. This is the aggregate
# frame rate: total frames / (UMI duration + real-robot duration).
export PICK_CUBE_DATASET_FREQUENCY=${DATASET_FREQUENCY:-20.158493127344187}
export PICK_CUBE_OBS_DOWNSAMPLE=3
export PICK_CUBE_ACTION_DOWNSAMPLE=3
export NUM_EPOCHS=${NUM_EPOCHS:-200}
export GRADIENT_ACCUMULATE_EVERY=${GRADIENT_ACCUMULATE_EVERY:-2}
export BATCH_SIZE=${BATCH_SIZE:-512}
export VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-128}
export PICK_CUBE_RUN_TAG=pick_cube_mix_umi20_real_all_1cam_with_state
exec "$script_dir/_pick_cube_train_common.sh"
