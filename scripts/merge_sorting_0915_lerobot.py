#!/usr/bin/env python3
"""Merge sorting_0915 into one trainable LeRobot v3 dataset.

The source has one left, right, and top parquet per episode.  This program
validates their shared timeline, writes one canonical LeRobot parquet per
episode, and packet-remuxes the three videos onto an exact integer-FPS time
base.  It never modifies or removes source data.
"""

from __future__ import annotations

import argparse
import os
import shutil
import uuid
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import datasets
import numpy as np
import pandas as pd

from lerobot.datasets.feature_utils import get_hf_features_from_features
from lerobot.datasets.video_utils import get_video_info

from sorting_0915_common import (
    ACTION_COLUMNS,
    CAMERAS,
    STATE_COLUMNS,
    column_matrix,
    discover_episodes,
    load_synced_episode,
    numeric_stats,
    video_path,
    write_json,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = SCRIPT_DIR.parent / "data"
DEFAULT_INPUT = DATA_ROOT / "umi_data_lerobot_preprocess" / "sorting_0915"
DEFAULT_OUTPUT = DATA_ROOT / "umi_data_lerobot_preprocess" / "sorting_0915_merged"
VECTOR_NAMES = tuple(
    f"{hand}.{name}" for hand in ("left", "right") for name in STATE_COLUMNS
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--task", default="pick_and_place")
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def output_indices(episode_index: int) -> tuple[int, int]:
    return episode_index // 1000, episode_index % 1000


def retime_video(source: Path, destination: Path, fps: int) -> dict[str, Any]:
    """Copy compressed packets while replacing PTS/DTS with exact 1/fps ticks."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    packet_count = 0
    with av.open(str(source), mode="r") as input_container:
        input_stream = input_container.streams.video[0]
        source_rate = input_stream.average_rate or input_stream.base_rate
        if source_rate is None or source_rate <= 0:
            raise ValueError(f"cannot determine video rate: {source}")
        source_frames = int(input_stream.frames or 0)
        with av.open(str(temporary), mode="w", format="mp4") as output_container:
            output_stream = output_container.add_stream_from_template(input_stream)
            output_time_base = Fraction(1, fps * 1000)
            output_stream.time_base = output_time_base
            for packet in input_container.demux(input_stream):
                if packet.dts is None:
                    continue
                old_time_base = packet.time_base
                if old_time_base is None:
                    raise ValueError(f"packet has no time base: {source}")
                pts_index = (
                    None
                    if packet.pts is None
                    else round(Fraction(packet.pts) * old_time_base * source_rate)
                )
                dts_index = round(
                    Fraction(packet.dts) * old_time_base * source_rate
                )
                packet.pts = None if pts_index is None else pts_index * 1000
                packet.dts = dts_index * 1000
                packet.duration = 1000
                packet.time_base = output_time_base
                packet.stream = output_stream
                output_container.mux(packet)
                packet_count += 1
    os.replace(temporary, destination)

    with av.open(str(destination), mode="r") as container:
        stream = container.streams.video[0]
        output_frames = int(stream.frames or 0)
        output_rate = stream.average_rate or stream.base_rate
        if output_rate != fps:
            raise AssertionError(
                f"retimed rate is {output_rate}, expected {fps}: {destination}"
            )
        if source_frames and output_frames != source_frames:
            raise AssertionError(
                f"retime changed frame count {source_frames}->{output_frames}: {source}"
            )
        return {
            "source_rate": float(source_rate),
            "output_rate": float(output_rate),
            "frames": output_frames,
            "packets": packet_count,
            "width": int(stream.width),
            "height": int(stream.height),
        }


def make_features(video_info: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (16,),
            "names": list(VECTOR_NAMES),
        },
        "action": {
            "dtype": "float32",
            "shape": (16,),
            "names": list(VECTOR_NAMES),
        },
        "source_timestamp": {
            "dtype": "float64",
            "shape": (1,),
            "names": None,
        },
    }
    for camera in CAMERAS:
        info = video_info[camera]
        features[f"observation.images.{camera}"] = {
            "dtype": "video",
            "shape": (int(info["video.height"]), int(info["video.width"]), 3),
            "names": ["height", "width", "channel"],
            "info": info,
        }
    features.update(
        {
            "timestamp": {"dtype": "float32", "shape": (1,), "names": None},
            "frame_index": {"dtype": "int64", "shape": (1,), "names": None},
            "episode_index": {"dtype": "int64", "shape": (1,), "names": None},
            "index": {"dtype": "int64", "shape": (1,), "names": None},
            "task_index": {"dtype": "int64", "shape": (1,), "names": None},
        }
    )
    return features


def write_episode_parquet(
    destination: Path,
    features: dict[str, dict[str, Any]],
    state: np.ndarray,
    action: np.ndarray,
    source_timestamps: np.ndarray,
    episode_index: int,
    global_start: int,
    fps: int,
) -> None:
    length = len(state)
    columns = {
        "observation.state": state,
        "action": action,
        "source_timestamp": source_timestamps,
        "timestamp": np.arange(length, dtype=np.float32) / np.float32(fps),
        "frame_index": np.arange(length, dtype=np.int64),
        "episode_index": np.full(length, episode_index, dtype=np.int64),
        "index": np.arange(global_start, global_start + length, dtype=np.int64),
        "task_index": np.zeros(length, dtype=np.int64),
    }
    non_video = {
        key: value for key, value in features.items() if value["dtype"] != "video"
    }
    hf_features = get_hf_features_from_features(non_video)
    dataset = datasets.Dataset.from_dict(columns, features=hf_features)
    destination.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(str(destination))


def prepare_output(path: Path, overwrite: bool) -> Path:
    path = path.expanduser().resolve()
    if path.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {path}; add --overwrite to replace it")
    temporary = path.with_name(f".{path.name}.partial-{uuid.uuid4().hex}")
    temporary.mkdir(parents=True)
    return temporary


def commit_output(temporary: Path, destination: Path, overwrite: bool) -> None:
    destination = destination.expanduser().resolve()
    if destination.exists():
        if not overwrite:
            raise FileExistsError(destination)
        shutil.rmtree(destination)
    os.replace(temporary, destination)


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    args.input_root = args.input_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    if args.input_root == args.output_root or args.input_root in args.output_root.parents:
        raise ValueError("output must not equal or be inside the source directory")
    episodes = discover_episodes(args.input_root, args.max_episodes)
    print(f"input: {args.input_root}")
    print(f"episodes: {len(episodes)}")
    print(f"output: {args.output_root}")
    if args.dry_run:
        total = 0
        for index, episode in enumerate(episodes, start=1):
            tables, _, _ = load_synced_episode(episode)
            total += len(tables["left"])
            print(f"validated {index}/{len(episodes)}: {episode.name} ({len(tables['left'])} frames)")
        print(f"validated total frames: {total}")
        return

    temporary = prepare_output(args.output_root, args.overwrite)
    global_index = 0
    episode_rows: list[dict[str, Any]] = []
    all_state: list[np.ndarray] = []
    all_action: list[np.ndarray] = []
    all_source_timestamp: list[np.ndarray] = []
    video_info: dict[str, dict[str, Any]] | None = None
    report_episodes: list[dict[str, Any]] = []

    try:
        for episode_index, episode in enumerate(episodes):
            tables, source_info, source_timestamps = load_synced_episode(episode)
            state = np.concatenate(
                [column_matrix(tables[hand], STATE_COLUMNS) for hand in ("left", "right")],
                axis=1,
            ).astype(np.float32, copy=False)
            action = np.concatenate(
                [column_matrix(tables[hand], ACTION_COLUMNS) for hand in ("left", "right")],
                axis=1,
            ).astype(np.float32, copy=False)
            length = len(state)
            chunk_index, file_index = output_indices(episode_index)
            data_path = (
                temporary
                / "data"
                / f"chunk-{chunk_index:03d}"
                / f"file-{file_index:03d}.parquet"
            )

            if video_info is None:
                video_info = {}
            video_report: dict[str, Any] = {}
            for camera in CAMERAS:
                destination = (
                    temporary
                    / "videos"
                    / f"observation.images.{camera}"
                    / f"chunk-{chunk_index:03d}"
                    / f"file-{file_index:03d}.mp4"
                )
                result = retime_video(video_path(episode, camera), destination, args.fps)
                if result["frames"] != length:
                    raise ValueError(
                        f"{episode.name}/{camera}: video={result['frames']}, parquet={length}"
                    )
                video_report[camera] = result
                if episode_index == 0:
                    video_info[camera] = get_video_info(destination)

            assert video_info is not None and len(video_info) == len(CAMERAS)
            features = make_features(video_info)
            write_episode_parquet(
                data_path,
                features,
                state,
                action,
                source_timestamps,
                episode_index,
                global_index,
                args.fps,
            )

            row: dict[str, Any] = {
                "episode_index": episode_index,
                "tasks": [args.task],
                "length": length,
                "data/chunk_index": chunk_index,
                "data/file_index": file_index,
                "dataset_from_index": global_index,
                "dataset_to_index": global_index + length,
            }
            for camera in CAMERAS:
                key = f"observation.images.{camera}"
                row[f"videos/{key}/chunk_index"] = chunk_index
                row[f"videos/{key}/file_index"] = file_index
                row[f"videos/{key}/from_timestamp"] = 0.0
                row[f"videos/{key}/to_timestamp"] = (length - 1) / args.fps
            episode_rows.append(row)
            all_state.append(state)
            all_action.append(action)
            all_source_timestamp.append(source_timestamps)
            report_episodes.append(
                {
                    "episode_index": episode_index,
                    "source_name": episode.name,
                    "frames": length,
                    "source_fps": source_info.get("fps"),
                    "video": video_report,
                }
            )
            global_index += length
            print(
                f"converted {episode_index + 1}/{len(episodes)}: "
                f"{episode.name} ({length} frames)",
                flush=True,
            )

        assert video_info is not None
        features = make_features(video_info)
        info = {
            "codebase_version": "v3.0",
            "fps": args.fps,
            "features": features,
            "total_episodes": len(episodes),
            "total_frames": global_index,
            "total_tasks": 1,
            "chunks_size": 1000,
            "data_files_size_in_mb": 100,
            "video_files_size_in_mb": 200,
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "robot_type": "umi_bimanual",
            "splits": {"train": f"0:{len(episodes)}"},
        }
        write_json(temporary / "meta" / "info.json", info)

        episode_columns = {
            key: [row[key] for row in episode_rows] for key in episode_rows[0]
        }
        episodes_dataset = datasets.Dataset.from_dict(episode_columns)
        episodes_path = temporary / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        episodes_path.parent.mkdir(parents=True, exist_ok=True)
        episodes_dataset.to_parquet(str(episodes_path))

        tasks = pd.DataFrame(
            {"task_index": [0]}, index=pd.Index([args.task], name="task")
        )
        tasks.to_parquet(temporary / "meta" / "tasks.parquet")

        state_all = np.concatenate(all_state, axis=0)
        action_all = np.concatenate(all_action, axis=0)
        source_timestamp_all = np.concatenate(all_source_timestamp)
        standard_timestamp = np.concatenate(
            [np.arange(len(value), dtype=np.float32) / args.fps for value in all_state]
        )
        stats = {
            "observation.state": numeric_stats(state_all),
            "action": numeric_stats(action_all),
            "source_timestamp": numeric_stats(source_timestamp_all),
            "timestamp": numeric_stats(standard_timestamp),
        }
        write_json(temporary / "meta" / "stats.json", stats)
        write_json(
            temporary / "conversion.json",
            {
                "conversion_complete": True,
                "source": str(args.input_root),
                "source_unchanged": True,
                "output_format": "LeRobot v3.0",
                "episodes": len(episodes),
                "frames": global_index,
                "fps": args.fps,
                "state_layout": list(VECTOR_NAMES),
                "action_layout": list(VECTOR_NAMES),
                "timestamp": (
                    "LeRobot timestamp is frame_index/fps per episode; "
                    "the synchronized measured clock is preserved as source_timestamp"
                ),
                "video_retime": (
                    "compressed H.264 packets copied without decoding; PTS/DTS replaced "
                    "by exact integer-fps timestamps; no frames added, removed, or reordered"
                ),
                "source_episodes": report_episodes,
            },
        )
        commit_output(temporary, args.output_root, args.overwrite)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    print(f"done: {args.output_root}")
    print(f"episodes={len(episodes)}, frames={global_index}, fps={args.fps}")


if __name__ == "__main__":
    main()
