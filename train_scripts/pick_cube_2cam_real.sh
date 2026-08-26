#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PICK_CUBE_SOURCE=real
export PICK_CUBE_TASK=umi_2cam
export PICK_CUBE_DATASET_PATH=${DATASET_PATH:-data/dataset_real_zarr/pick_cube_2cam/dataset.zarr.zip}
export PICK_CUBE_IGNORE_PROPRIOCEPTION=false
export PICK_CUBE_DATASET_FREQUENCY=20
export PICK_CUBE_OBS_DOWNSAMPLE=1
export PICK_CUBE_ACTION_DOWNSAMPLE=1
export NUM_EPOCHS=${NUM_EPOCHS:-200}
export GRADIENT_ACCUMULATE_EVERY=${GRADIENT_ACCUMULATE_EVERY:-4}
export BATCH_SIZE=${BATCH_SIZE:-256}
export VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-128}
export PICK_CUBE_RUN_TAG=pick_cube_real_2cam_with_state
exec "$script_dir/_pick_cube_train_common.sh"
