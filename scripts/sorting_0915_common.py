#!/usr/bin/env python3
"""Shared validation and conversion helpers for the sorting_0915 dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


CAMERAS = ("left", "right", "top")
HANDS = ("left", "right")
POSE_COLUMNS = ("tx", "ty", "tz", "qw", "qx", "qy", "qz")
STATE_COLUMNS = (*POSE_COLUMNS, "gripper")
ACTION_COLUMNS = tuple(f"action.{name}" for name in STATE_COLUMNS)
PIPELINE = "sorting_0915_bimanual_crop_mask_sync_v1"


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def discover_episodes(input_root: Path, max_episodes: int | None = None) -> list[Path]:
    episodes = sorted(
        path for path in input_root.glob("*_lerobot") if path.is_dir()
    )
    if not episodes:
        raise FileNotFoundError(f"no *_lerobot episode directories under {input_root}")
    if max_episodes is not None:
        if max_episodes <= 0:
            raise ValueError("--max-episodes must be positive")
        episodes = episodes[:max_episodes]
    return episodes


def parquet_path(episode: Path, stream: str) -> Path:
    return episode / "data" / stream / "chunk-000" / "file_000.parquet"


def video_path(episode: Path, camera: str) -> Path:
    return (
        episode
        / "videos"
        / f"observation.images.{camera}"
        / "chunk-000"
        / "file_000.mp4"
    )


def column_matrix(table: pa.Table, names: tuple[str, ...]) -> np.ndarray:
    return np.column_stack(
        [np.asarray(table[name].to_numpy(), dtype=np.float32) for name in names]
    ).astype(np.float32, copy=False)


def _validate_table(
    episode: Path,
    stream: str,
    table: pa.Table,
    reference_timestamps: np.ndarray | None,
) -> np.ndarray:
    required = {"frame_index", "timestamp"}
    if stream in HANDS:
        required.update(STATE_COLUMNS)
        required.update(ACTION_COLUMNS)
    missing = sorted(required - set(table.column_names))
    if missing:
        raise ValueError(f"{episode.name}/{stream}: missing columns {missing}")
    if len(table) == 0:
        raise ValueError(f"{episode.name}/{stream}: empty parquet")

    frame_index = np.asarray(table["frame_index"].to_numpy(), dtype=np.int64)
    if not np.array_equal(frame_index, np.arange(len(table), dtype=np.int64)):
        raise ValueError(
            f"{episode.name}/{stream}: frame_index is not continuous from zero"
        )
    timestamps = np.asarray(table["timestamp"].to_numpy(), dtype=np.float64)
    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
        raise ValueError(
            f"{episode.name}/{stream}: timestamp must be finite and increasing"
        )
    if reference_timestamps is not None and not np.array_equal(
        timestamps, reference_timestamps
    ):
        max_error = (
            float(np.max(np.abs(timestamps - reference_timestamps)))
            if len(timestamps) == len(reference_timestamps)
            else float("inf")
        )
        raise ValueError(
            f"{episode.name}/{stream}: timeline differs from left; "
            f"length={len(timestamps)}/{len(reference_timestamps)}, "
            f"max_error={max_error}"
        )
    for name in required - {"frame_index", "timestamp"}:
        values = np.asarray(table[name].to_numpy())
        if not np.isfinite(values).all():
            raise ValueError(f"{episode.name}/{stream}.{name}: NaN or Inf")
    if stream in HANDS:
        next_index = np.minimum(np.arange(len(table)) + 1, len(table) - 1)
        for state_name, action_name in zip(STATE_COLUMNS, ACTION_COLUMNS):
            state = np.asarray(table[state_name].to_numpy(), dtype=np.float64)
            action = np.asarray(table[action_name].to_numpy(), dtype=np.float64)
            if not np.allclose(action, state[next_index], rtol=0.0, atol=2e-6):
                error = float(np.max(np.abs(action - state[next_index])))
                raise ValueError(
                    f"{episode.name}/{stream}: {action_name} is not the next "
                    f"{state_name}; max_error={error}"
                )
        quaternion = column_matrix(table, POSE_COLUMNS[3:])
        if not np.allclose(
            np.linalg.norm(quaternion, axis=1), 1.0, rtol=0.0, atol=2e-6
        ):
            raise ValueError(f"{episode.name}/{stream}: quaternion is not normalized")
    return timestamps


def load_synced_episode(
    episode: Path,
    cameras: tuple[str, ...] = CAMERAS,
) -> tuple[dict[str, pa.Table], dict[str, Any], np.ndarray]:
    if not cameras:
        raise ValueError("at least one camera/stream is required")
    info = load_json(episode / "meta" / "info.json")
    declared = info.get("sorting_0915_preprocess", {})
    if declared.get("pipeline") != PIPELINE:
        raise ValueError(
            f"{episode.name}: expected preprocess pipeline {PIPELINE!r}, "
            f"got {declared.get('pipeline')!r}"
        )
    if info.get("pose_frame") != "first_frame_delta":
        raise ValueError(
            f"{episode.name}: expected pose_frame='first_frame_delta', "
            f"got {info.get('pose_frame')!r}"
        )

    tables: dict[str, pa.Table] = {}
    reference: np.ndarray | None = None
    for stream in cameras:
        path = parquet_path(episode, stream)
        if not path.is_file():
            raise FileNotFoundError(f"{episode.name}: missing {path.relative_to(episode)}")
        table = pq.read_table(path).combine_chunks()
        timestamps = _validate_table(episode, stream, table, reference)
        if reference is None:
            reference = timestamps
        tables[stream] = table
        video = video_path(episode, stream)
        if not video.is_file():
            raise FileNotFoundError(
                f"{episode.name}: missing {video.relative_to(episode)}"
            )

    assert reference is not None
    frame_count = len(reference)
    for stream in cameras:
        if len(tables[stream]) != frame_count:
            raise ValueError(
                f"{episode.name}: {stream} has {len(tables[stream])} rows, "
                f"left has {frame_count}"
            )
        declared_frames = info.get(f"total_frames_{stream}")
        if declared_frames is not None and int(declared_frames) != frame_count:
            raise ValueError(
                f"{episode.name}: info total_frames_{stream}={declared_frames}, "
                f"parquet={frame_count}"
            )
    return tables, info, reference


def numeric_stats(values: np.ndarray) -> dict[str, list[float] | list[int]]:
    array = np.asarray(values)
    if array.ndim == 1:
        array = array[:, None]
    if len(array) == 0 or not np.isfinite(array).all():
        raise ValueError("statistics input must be non-empty and finite")
    work = array.astype(np.float64, copy=False)
    result: dict[str, list[float] | list[int]] = {
        "min": np.min(work, axis=0).tolist(),
        "max": np.max(work, axis=0).tolist(),
        "mean": np.mean(work, axis=0).tolist(),
        "std": np.std(work, axis=0).tolist(),
        "count": [int(len(work))],
    }
    for quantile in (0.01, 0.10, 0.50, 0.90, 0.99):
        result[f"q{int(quantile * 100):02d}"] = np.quantile(
            work, quantile, axis=0
        ).tolist()
    return result
