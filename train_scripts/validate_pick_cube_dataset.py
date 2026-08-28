#!/usr/bin/env python3
"""Fail fast unless a pick_cube Zarr follows the canonical TCP contract."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import zarr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs

register_codecs()


EXPECTED_AXES = {"+X": "forward", "+Y": "left", "+Z": "up"}


def validate_dataset(path: Path, camera_count: int) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"dataset not found: {path}")
    with zarr.ZipStore(str(path), mode="r") as store:
        root = zarr.open_group(store=store, mode="r")
        convention = dict(root.attrs.get("coordinate_convention", {}))
        errors: list[str] = []
        expected = {
            "name": "table_tcp_absolute",
            "frame": "fixed_table_frame",
            "table_axes": EXPECTED_AXES,
            "axis_map_applied": False,
            "episode_rezero_applied": False,
        }
        for key, value in expected.items():
            if convention.get(key) != value:
                errors.append(
                    f"coordinate_convention.{key}={convention.get(key)!r}, "
                    f"expected {value!r}"
                )

        data = root.get("data")
        if data is None:
            errors.append("missing data group")
        else:
            required = {
                "robot0_eef_pos",
                "robot0_eef_rot_axis_angle",
                "robot0_gripper_width",
                "camera0_rgb",
            }
            if camera_count == 2:
                required.add("camera1_rgb")
            missing = sorted(required - set(data.array_keys()))
            if missing:
                errors.append(f"missing arrays: {missing}")
            present = [key for key in required if key in data]
            lengths = {key: int(data[key].shape[0]) for key in present}
            if lengths and (min(lengths.values()) <= 0 or len(set(lengths.values())) != 1):
                errors.append(f"array lengths are empty or inconsistent: {lengths}")

        if errors:
            detail = "\n  - ".join(errors)
            raise ValueError(
                f"{path} is not a canonical pick_cube table-TCP dataset:\n  - {detail}"
            )
    return convention


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--camera-count", type=int, choices=(1, 2), required=True)
    args = parser.parse_args()
    convention = validate_dataset(args.dataset.resolve(), args.camera_count)
    print(
        "Coordinate contract OK: "
        f"{convention['name']}, axes={convention['table_axes']}, "
        "model pose_repr=relative"
    )


if __name__ == "__main__":
    main()
