#!/usr/bin/env python3
"""Convert synchronized sorting_0915 episodes to a three-camera bimanual Zarr.

The source remains untouched.  robot0 is the left hand.  robot1 is the right
hand transformed into the left-hand frame at source frame zero with the
per-episode T_left_right transform stored in meta/info.json.
"""

from __future__ import annotations

import argparse
import os
import shutil
import uuid
import zipfile
from pathlib import Path
from typing import Any

import av
import cv2
import imagecodecs.numcodecs
import numpy as np
import zarr
from scipy.spatial.transform import Rotation

from sorting_0915_common import (
    ACTION_COLUMNS,
    CAMERAS,
    POSE_COLUMNS,
    STATE_COLUMNS,
    column_matrix,
    discover_episodes,
    load_synced_episode,
    video_path,
    write_json,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = SCRIPT_DIR.parent / "data"
DEFAULT_INPUT = DATA_ROOT / "umi_data_lerobot_preprocess" / "sorting_0915"
DEFAULT_OUTPUT = DATA_ROOT / "dataset_umi_zarr" / "sorting_0915_3cam" / "dataset.zarr.zip"
CAMERA_MAPPING = {
    "left": "camera0_rgb",
    "right": "camera1_rgb",
    "top": "camera2_rgb",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--jpegxl-level", type=int, default=99)
    parser.add_argument("--gripper-open-metres", type=float, default=0.09)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def quaternion_wxyz_to_matrix(values: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(values, dtype=np.float64)
    if quaternion.ndim != 2 or quaternion.shape[1] != 4:
        raise ValueError(f"expected quaternion [N,4], got {quaternion.shape}")
    norms = np.linalg.norm(quaternion, axis=1)
    if not np.isfinite(quaternion).all() or np.any(norms < 1e-8):
        raise ValueError("quaternion contains NaN/Inf or has zero norm")
    normalized = quaternion / norms[:, None]
    return Rotation.from_quat(normalized[:, [1, 2, 3, 0]]).as_matrix()


def poses_to_matrices(values: np.ndarray) -> np.ndarray:
    pose = np.asarray(values, dtype=np.float64)
    matrices = np.broadcast_to(np.eye(4), (len(pose), 4, 4)).copy()
    matrices[:, :3, :3] = quaternion_wxyz_to_matrix(pose[:, 3:7])
    matrices[:, :3, 3] = pose[:, :3]
    return matrices


def transform_from_metadata(info: dict[str, Any]) -> np.ndarray:
    relation = info.get("hands", {}).get("hands_relative_pose_frame0", {})
    if relation.get("convention") != "T_left_right":
        raise ValueError(
            "meta/info.json must contain hands_relative_pose_frame0 with "
            "convention='T_left_right'"
        )
    vector = np.asarray(
        [relation[name] for name in POSE_COLUMNS], dtype=np.float64
    )[None, :]
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quaternion_wxyz_to_matrix(vector[:, 3:7])[0]
    transform[:3, 3] = vector[0, :3]
    return transform


def matrices_to_pose(matrices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(matrices, dtype=np.float64)
    position = values[:, :3, 3].astype(np.float32)
    rotvec = Rotation.from_matrix(values[:, :3, :3]).as_rotvec().astype(np.float32)
    return position, rotvec


def gripper_to_metres(values: np.ndarray, open_metres: float) -> tuple[np.ndarray, int]:
    source = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    clipped = int(np.count_nonzero((source < 0.0) | (source > 1.0)))
    result = np.clip(source, 0.0, 1.0) * open_metres
    return result.astype(np.float32), clipped


def pose_columns(table: Any, action: bool = False) -> np.ndarray:
    names = ACTION_COLUMNS if action else STATE_COLUMNS
    return column_matrix(table, names)


def common_frame_pose(
    pose: np.ndarray, transform: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrices = poses_to_matrices(pose[:, :7])
    if transform is not None:
        matrices = np.einsum("ij,njk->nik", transform, matrices)
    position, rotvec = matrices_to_pose(matrices)
    return position, rotvec, matrices


def repeat_boundary_pose(
    position: np.ndarray, rotvec: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    pose = np.concatenate((position, rotvec), axis=1).astype(np.float32)
    return (
        np.repeat(pose[:1], len(pose), axis=0),
        np.repeat(pose[-1:], len(pose), axis=0),
    )


def load_lowdim(
    episodes: list[Path], open_metres: float
) -> tuple[dict[str, np.ndarray], np.ndarray, list[dict[str, Any]]]:
    pieces: dict[str, list[np.ndarray]] = {
        "robot0_eef_pos": [],
        "robot0_eef_rot_axis_angle": [],
        "robot0_gripper_width": [],
        "robot0_demo_start_pose": [],
        "robot0_demo_end_pose": [],
        "robot1_eef_pos": [],
        "robot1_eef_rot_axis_angle": [],
        "robot1_gripper_width": [],
        "robot1_demo_start_pose": [],
        "robot1_demo_end_pose": [],
        "action": [],
        "timestamp": [],
        "episode_frame_index": [],
    }
    episode_ends: list[int] = []
    reports: list[dict[str, Any]] = []
    cumulative = 0

    for episode_index, episode in enumerate(episodes):
        tables, info, timestamps = load_synced_episode(episode)
        left_state = pose_columns(tables["left"])
        right_state = pose_columns(tables["right"])
        left_action = pose_columns(tables["left"], action=True)
        right_action = pose_columns(tables["right"], action=True)
        left_to_right0 = transform_from_metadata(info)

        left_pos, left_rot, _ = common_frame_pose(left_state, None)
        right_pos, right_rot, _ = common_frame_pose(right_state, left_to_right0)
        left_action_pos, left_action_rot, _ = common_frame_pose(left_action, None)
        right_action_pos, right_action_rot, _ = common_frame_pose(
            right_action, left_to_right0
        )
        left_gripper, left_clipped = gripper_to_metres(
            left_state[:, 7], open_metres
        )
        right_gripper, right_clipped = gripper_to_metres(
            right_state[:, 7], open_metres
        )
        left_action_gripper, left_action_clipped = gripper_to_metres(
            left_action[:, 7], open_metres
        )
        right_action_gripper, right_action_clipped = gripper_to_metres(
            right_action[:, 7], open_metres
        )
        action = np.concatenate(
            (
                left_action_pos,
                left_action_rot,
                left_action_gripper,
                right_action_pos,
                right_action_rot,
                right_action_gripper,
            ),
            axis=1,
        ).astype(np.float32)
        left_start, left_end = repeat_boundary_pose(left_pos, left_rot)
        right_start, right_end = repeat_boundary_pose(right_pos, right_rot)

        episode_values = {
            "robot0_eef_pos": left_pos,
            "robot0_eef_rot_axis_angle": left_rot,
            "robot0_gripper_width": left_gripper,
            "robot0_demo_start_pose": left_start,
            "robot0_demo_end_pose": left_end,
            "robot1_eef_pos": right_pos,
            "robot1_eef_rot_axis_angle": right_rot,
            "robot1_gripper_width": right_gripper,
            "robot1_demo_start_pose": right_start,
            "robot1_demo_end_pose": right_end,
            "action": action,
            "timestamp": timestamps[:, None].astype(np.float64),
            "episode_frame_index": np.arange(len(timestamps), dtype=np.int64)[:, None],
        }
        for key, value in episode_values.items():
            pieces[key].append(value)

        cumulative += len(timestamps)
        episode_ends.append(cumulative)
        dt = np.diff(timestamps)
        reports.append(
            {
                "episode_index": episode_index,
                "source_name": episode.name,
                "frames": len(timestamps),
                "timestamp_start": float(timestamps[0]),
                "timestamp_end": float(timestamps[-1]),
                "median_dt_sec": float(np.median(dt)) if len(dt) else None,
                "source_fps": info.get("fps"),
                "T_left_right": left_to_right0.tolist(),
                "gripper_clipped_values": {
                    "left_state": left_clipped,
                    "right_state": right_clipped,
                    "left_action": left_action_clipped,
                    "right_action": right_action_clipped,
                },
            }
        )
        print(
            f"loaded lowdim {episode_index + 1}/{len(episodes)}: "
            f"{episode.name} ({len(timestamps)} frames)",
            flush=True,
        )

    arrays = {key: np.concatenate(value, axis=0) for key, value in pieces.items()}
    for key, value in arrays.items():
        if not np.isfinite(value).all():
            raise ValueError(f"converted lowdim contains NaN/Inf: {key}")
    return arrays, np.asarray(episode_ends, dtype=np.int64), reports


def create_store(
    directory: Path,
    arrays: dict[str, np.ndarray],
    episode_ends: np.ndarray,
    episodes: list[Path],
    height: int,
    width: int,
    jpegxl_level: int,
    open_metres: float,
) -> dict[str, zarr.Array]:
    imagecodecs.numcodecs.register_codecs()
    root = zarr.open_group(str(directory), mode="w")
    root.attrs["coordinate_convention"] = {
        "name": "left_frame0_common",
        "frame": "per-episode left-hand pose at source frame zero",
        "input_pose_frame": "first_frame_delta",
        "robot0": "left; T_common_robot0(t) = T_left_delta(t)",
        "robot1": (
            "right; T_common_robot1(t) = T_left_right(frame0) "
            "@ T_right_delta(t)"
        ),
        "T_left_right_source": "each source episode meta/info.json",
        "not_fixed_table_absolute": True,
    }
    root.attrs["camera_mapping"] = {
        output: f"observation.images.{source}"
        for source, output in CAMERA_MAPPING.items()
    }
    root.attrs["trajectory_layout"] = {
        "robot0": "left",
        "robot1": "right transformed into left-frame0 common frame",
        "position": "metres",
        "rotation": "rotation vector, radians",
        "gripper": f"source [0,1] clipped and mapped to [0,{open_metres}] metres",
        "action": (
            "[robot0 target xyz, target rotvec, target gripper metres, "
            "robot1 target xyz, target rotvec, target gripper metres]"
        ),
    }
    root.attrs["source"] = str(episodes[0].parent.resolve())
    root.attrs["source_episodes"] = [episode.name for episode in episodes]
    root.attrs["source_unchanged"] = True

    data = root.require_group("data")
    meta = root.require_group("meta")
    for key, value in arrays.items():
        chunks = (min(255, len(value)), *value.shape[1:])
        data.array(key, value, chunks=chunks, compressor=None)
    meta.array(
        "episode_ends",
        episode_ends,
        chunks=(max(1, len(episode_ends)),),
        compressor=None,
    )

    compressor = imagecodecs.numcodecs.Jpegxl(
        level=jpegxl_level, lossless=False, numthreads=1
    )
    total = int(episode_ends[-1])
    return {
        output: data.zeros(
            output,
            shape=(total, height, width, 3),
            chunks=(1, height, width, 3),
            dtype=np.uint8,
            compressor=compressor,
        )
        for output in CAMERA_MAPPING.values()
    }


def write_camera(
    episodes: list[Path],
    camera: str,
    destination: zarr.Array,
    episode_ends: np.ndarray,
) -> None:
    output_index = 0
    expected_source_shape: tuple[int, int, int] | None = None
    output_width = int(destination.shape[2])
    output_height = int(destination.shape[1])
    for episode_index, episode in enumerate(episodes):
        start = 0 if episode_index == 0 else int(episode_ends[episode_index - 1])
        end = int(episode_ends[episode_index])
        expected = end - start
        decoded = 0
        with av.open(str(video_path(episode, camera)), mode="r") as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            source_shape = (int(stream.height), int(stream.width), 3)
            if expected_source_shape is None:
                expected_source_shape = source_shape
            elif source_shape != expected_source_shape:
                raise ValueError(
                    f"{episode.name}/{camera}: video shape changed from "
                    f"{expected_source_shape} to {source_shape}"
                )
            interpolation = (
                cv2.INTER_AREA
                if output_width <= stream.width and output_height <= stream.height
                else cv2.INTER_LINEAR
            )
            for frame in container.decode(stream):
                if decoded >= expected:
                    raise ValueError(
                        f"{episode.name}/{camera}: video has more frames than parquet"
                    )
                image = frame.to_ndarray(format="rgb24")
                image = cv2.resize(
                    image,
                    (output_width, output_height),
                    interpolation=interpolation,
                )
                destination[output_index] = np.ascontiguousarray(image, dtype=np.uint8)
                decoded += 1
                output_index += 1
        if decoded != expected:
            raise ValueError(
                f"{episode.name}/{camera}: video={decoded}, parquet={expected}"
            )
        print(
            f"  {camera}: episode {episode_index + 1}/{len(episodes)}, "
            f"cumulative={output_index}",
            flush=True,
        )


def zip_store(source: Path, destination: Path) -> None:
    with zarr.ZipStore(
        str(destination), mode="w", compression=zipfile.ZIP_STORED, allowZip64=True
    ) as output_store:
        zarr.copy_store(
            zarr.DirectoryStore(str(source)), output_store, if_exists="replace"
        )


def validate_output(
    path: Path,
    episode_ends: np.ndarray,
    height: int,
    width: int,
) -> None:
    total = int(episode_ends[-1])
    expected_lowdim = {
        "robot0_eef_pos": (total, 3),
        "robot0_eef_rot_axis_angle": (total, 3),
        "robot0_gripper_width": (total, 1),
        "robot0_demo_start_pose": (total, 6),
        "robot0_demo_end_pose": (total, 6),
        "robot1_eef_pos": (total, 3),
        "robot1_eef_rot_axis_angle": (total, 3),
        "robot1_gripper_width": (total, 1),
        "robot1_demo_start_pose": (total, 6),
        "robot1_demo_end_pose": (total, 6),
        "action": (total, 14),
        "timestamp": (total, 1),
        "episode_frame_index": (total, 1),
    }
    with zarr.ZipStore(str(path), mode="r") as store:
        root = zarr.group(store=store)
        if not np.array_equal(root["meta/episode_ends"][:], episode_ends):
            raise AssertionError("episode_ends changed while writing")
        convention = root.attrs.get("coordinate_convention", {})
        if convention.get("name") != "left_frame0_common":
            raise AssertionError("missing coordinate convention receipt")
        for key, shape in expected_lowdim.items():
            value = root[f"data/{key}"]
            if value.shape != shape:
                raise AssertionError(f"{key}: shape={value.shape}, expected={shape}")
            if not np.isfinite(value[:]).all():
                raise AssertionError(f"{key}: contains NaN/Inf")
        for key in CAMERA_MAPPING.values():
            value = root[f"data/{key}"]
            expected_shape = (total, height, width, 3)
            if value.shape != expected_shape or value.dtype != np.uint8:
                raise AssertionError(
                    f"{key}: shape/dtype={value.shape}/{value.dtype}, "
                    f"expected={expected_shape}/uint8"
                )
            for index in (0, total - 1):
                image = value[index]
                if image.shape != (height, width, 3) or not np.any(image):
                    raise AssertionError(f"{key}[{index}] is empty or unreadable")


def main() -> None:
    args = parse_args()
    args.input_root = args.input_root.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if args.height <= 0 or args.width <= 0:
        raise ValueError("--height and --width must be positive")
    if not 1 <= args.jpegxl_level <= 100:
        raise ValueError("--jpegxl-level must be in [1,100]")
    if not np.isfinite(args.gripper_open_metres) or args.gripper_open_metres <= 0:
        raise ValueError("--gripper-open-metres must be positive and finite")
    if args.input_root == args.output or args.input_root in args.output.parents:
        raise ValueError("output must not equal or be inside the source directory")
    episodes = discover_episodes(args.input_root, args.max_episodes)
    print(f"input: {args.input_root}")
    print(f"episodes: {len(episodes)}")
    print(f"output: {args.output}")

    if args.dry_run:
        total = 0
        for index, episode in enumerate(episodes, start=1):
            tables, info, _ = load_synced_episode(episode)
            transform_from_metadata(info)
            total += len(tables["left"])
            print(f"validated {index}/{len(episodes)}: {episode.name} ({len(tables['left'])} frames)")
        print(f"validated total frames: {total}")
        return

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"output already exists: {args.output}; add --overwrite to replace it"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    work = args.output.parent / f".sorting_0915_zarr.partial-{uuid.uuid4().hex}"
    directory = work / "dataset.zarr"
    temporary_zip = work / "dataset.zarr.zip"
    work.mkdir()
    try:
        arrays, episode_ends, reports = load_lowdim(
            episodes, args.gripper_open_metres
        )
        image_arrays = create_store(
            directory,
            arrays,
            episode_ends,
            episodes,
            args.height,
            args.width,
            args.jpegxl_level,
            args.gripper_open_metres,
        )
        for camera, output_key in CAMERA_MAPPING.items():
            print(f"writing {camera} -> data/{output_key}", flush=True)
            write_camera(episodes, camera, image_arrays[output_key], episode_ends)
        zip_store(directory, temporary_zip)
        validate_output(temporary_zip, episode_ends, args.height, args.width)

        if args.output.exists():
            if not args.overwrite:
                raise FileExistsError(args.output)
            args.output.unlink()
        os.replace(temporary_zip, args.output)

        valid_dt = [row["median_dt_sec"] for row in reports if row["median_dt_sec"]]
        observed_fps = float(1.0 / np.median(valid_dt)) if valid_dt else None
        report_path = args.output.with_suffix(args.output.suffix + ".conversion.json")
        write_json(
            report_path,
            {
                "conversion_complete": True,
                "source": str(args.input_root),
                "source_unchanged": True,
                "output": str(args.output),
                "output_format": "Data-Scaling-Laws UMI replay-buffer Zarr v2 ZIP",
                "episodes": len(episodes),
                "frames": int(episode_ends[-1]),
                "episode_ends": episode_ends.tolist(),
                "image_shape": [args.height, args.width, 3],
                "image_transform": "direct resize of already cropped/masked synchronized RGB",
                "camera_mapping": {
                    output: f"observation.images.{source}"
                    for source, output in CAMERA_MAPPING.items()
                },
                "coordinate_convention": {
                    "name": "left_frame0_common",
                    "robot0": "left first-frame-delta pose",
                    "robot1": "T_left_right(frame0) @ right first-frame-delta pose",
                    "warning": "not a fixed-table absolute TCP coordinate frame",
                },
                "action": (
                    "14D next-frame absolute target on the shared episode timeline: "
                    "left xyz+rotvec+gripper(m), right xyz+rotvec+gripper(m)"
                ),
                "timestamp": "original synchronized source timestamp retained in data/timestamp",
                "observed_fps": observed_fps,
                "recommended_training": {
                    "dataset_frequeny": observed_fps,
                    "obs_down_sample_steps": 1,
                    "action_down_sample_steps": 1,
                },
                "source_episodes": reports,
            },
        )
        (args.output.parent / "count.txt").write_text(
            f"{len(episodes)}\n", encoding="utf-8"
        )
    finally:
        shutil.rmtree(work, ignore_errors=True)

    print(f"done: {args.output}")
    print(f"episodes={len(episodes)}, frames={int(episode_ends[-1])}")


if __name__ == "__main__":
    main()
