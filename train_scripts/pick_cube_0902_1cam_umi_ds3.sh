#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PICK_CUBE_SOURCE=umi_0902
export PICK_CUBE_TASK=umi
export PICK_CUBE_DATASET_PATH=${DATASET_PATH:-data/dataset_umi_zarr/pick_cube_0902_1cam/dataset.zarr.zip}
export PICK_CUBE_IGNORE_PROPRIOCEPTION=false
export PICK_CUBE_DATASET_FREQUENCY=${DATASET_FREQUENCY:-19.99983038897112}
export PICK_CUBE_OBS_DOWNSAMPLE=3
export PICK_CUBE_ACTION_DOWNSAMPLE=3
export NUM_EPOCHS=${NUM_EPOCHS:-200}
export GRADIENT_ACCUMULATE_EVERY=${GRADIENT_ACCUMULATE_EVERY:-2}
export BATCH_SIZE=${BATCH_SIZE:-512}
export VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-128}
export PICK_CUBE_RUN_TAG=pick_cube_0902_umi_1cam_ds3_with_state
exec "$script_dir/_pick_cube_train_common.sh"
