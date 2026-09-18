#!/usr/bin/env python3
"""Fail fast unless a sorting_0915 bimanual Zarr matches the training scripts."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import zarr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs

register_codecs()


LOWDIM_SHAPES = {
    "robot0_eef_pos": (3,),
    "robot0_eef_rot_axis_angle": (3,),
    "robot0_gripper_width": (1,),
    "robot0_demo_start_pose": (6,),
    "robot0_demo_end_pose": (6,),
    "robot1_eef_pos": (3,),
    "robot1_eef_rot_axis_angle": (3,),
    "robot1_gripper_width": (1,),
    "robot1_demo_start_pose": (6,),
    "robot1_demo_end_pose": (6,),
    "action": (14,),
    "timestamp": (1,),
    "episode_frame_index": (1,),
}


def validate_dataset(path: Path, camera_count: int) -> tuple[int, int]:
    if not path.is_file():
        raise FileNotFoundError(f"dataset not found: {path}")

    with zarr.ZipStore(str(path), mode="r") as store:
        root = zarr.open_group(store=store, mode="r")
        errors: list[str] = []
        convention = dict(root.attrs.get("coordinate_convention", {}))
        if convention.get("name") != "left_frame0_common":
            errors.append(
                "coordinate_convention.name="
                f"{convention.get('name')!r}, expected 'left_frame0_common'"
            )

        data = root.get("data")
        meta = root.get("meta")
        if data is None:
            errors.append("missing data group")
        if meta is None or "episode_ends" not in meta:
            errors.append("missing meta/episode_ends")
        if data is None or errors:
            raise ValueError(f"{path} is not a sorting_0915 Zarr: {errors}")

        required_cameras = {f"camera{index}_rgb" for index in range(camera_count)}
        required = set(LOWDIM_SHAPES) | required_cameras
        missing = sorted(required - set(data.array_keys()))
        if missing:
            errors.append(f"missing arrays: {missing}")

        lengths: dict[str, int] = {}
        for key, trailing_shape in LOWDIM_SHAPES.items():
            if key not in data:
                continue
            value = data[key]
            if tuple(value.shape[1:]) != trailing_shape:
                errors.append(
                    f"{key}: shape={value.shape}, expected [N,{trailing_shape}]"
                )
            lengths[key] = int(value.shape[0])

        for key in required_cameras:
            if key not in data:
                continue
            value = data[key]
            if tuple(value.shape[1:]) != (224, 224, 3) or value.dtype != np.uint8:
                errors.append(
                    f"{key}: shape/dtype={value.shape}/{value.dtype}, "
                    "expected [N,224,224,3]/uint8"
                )
            lengths[key] = int(value.shape[0])

        if lengths and (min(lengths.values()) <= 0 or len(set(lengths.values())) != 1):
            errors.append(f"array lengths are empty or inconsistent: {lengths}")

        episode_count = 0
        total = next(iter(lengths.values())) if lengths else 0
        if meta is not None and "episode_ends" in meta:
            episode_ends = meta["episode_ends"][:]
            episode_count = int(len(episode_ends))
            if episode_count == 0:
                errors.append("meta/episode_ends is empty")
            elif lengths and int(episode_ends[-1]) != total:
                errors.append(
                    f"episode_ends[-1]={int(episode_ends[-1])}, expected {total}"
                )

        if errors:
            detail = "\n  - ".join(errors)
            raise ValueError(
                f"{path} is not a valid sorting_0915 bimanual dataset:\n  - {detail}"
            )
    return episode_count, total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--camera-count", type=int, choices=(2, 3), required=True)
    args = parser.parse_args()
    episodes, frames = validate_dataset(args.dataset.resolve(), args.camera_count)
    print(
        "sorting_0915 dataset OK: "
        f"{episodes} episodes, {frames} frames, {args.camera_count} cameras"
    )


if __name__ == "__main__":
    main()
