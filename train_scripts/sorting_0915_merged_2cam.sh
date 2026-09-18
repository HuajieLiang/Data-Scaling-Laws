#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export SORTING_LEROBOT_SOURCE_PATH=${LEROBOT_SOURCE_PATH:-data/umi_data_lerobot_preprocess/sorting_0915_merged_2cam}
export SORTING_DATASET_PATH=${DATASET_PATH:-data/dataset_umi_zarr/sorting_0915_2cam/dataset.zarr.zip}
export SORTING_CAMERA_COUNT=2
export SORTING_DATASET_FREQUENCY=${DATASET_FREQUENCY:-62.46062248913964}
export SORTING_OBS_DOWNSAMPLE=${OBS_DOWNSAMPLE:-1}
export SORTING_ACTION_DOWNSAMPLE=${ACTION_DOWNSAMPLE:-1}
export SORTING_RUN_TAG=sorting_0915_merged_2cam_bimanual
exec "$script_dir/_sorting_0915_train_common.sh"
