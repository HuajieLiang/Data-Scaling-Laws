#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PICK_CUBE_SOURCE=umi
export PICK_CUBE_TASK=umi
export PICK_CUBE_DATASET_PATH=${DATASET_PATH:-data/dataset_umi_zarr/pick_cube_trim2s_1cam/dataset.zarr.zip}
export PICK_CUBE_IGNORE_PROPRIOCEPTION=true
export PICK_CUBE_DATASET_FREQUENCY=23.12959389342191
export PICK_CUBE_OBS_DOWNSAMPLE=3
export PICK_CUBE_ACTION_DOWNSAMPLE=3
export NUM_EPOCHS=${NUM_EPOCHS:-200}
export GRADIENT_ACCUMULATE_EVERY=${GRADIENT_ACCUMULATE_EVERY:-2}
export BATCH_SIZE=${BATCH_SIZE:-512}
export VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-128}
export PICK_CUBE_RUN_TAG=pick_cube_trim2s_umi_1cam_no_state
exec "$script_dir/_pick_cube_train_common.sh"
