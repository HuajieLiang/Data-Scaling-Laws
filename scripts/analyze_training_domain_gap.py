#!/usr/bin/env python3
"""Diagnose UMI/real training-domain gaps from data and training logs.

The analysis intentionally separates three effects:

1. image appearance/camera geometry,
2. relative action and trajectory support,
3. optimization history (including the step-based LR schedule).

It reads the two pick-cube Zarr datasets without modifying them and writes a
compact JSON report plus CSV tables under the cross-evaluation output folder.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import pickle
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import zarr
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import rankdata, wasserstein_distance


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs  # noqa: E402
from umi.common.pose_util import mat_to_rot6d  # noqa: E402


register_codecs()

DATASETS = {
    "umi": {
        "path": PROJECT_ROOT / "data/dataset_umi_zarr/pick_cube_1cam/dataset.zarr.zip",
        "frequency_hz": 23.12959389342191,
    },
    "real": {
        "path": PROJECT_ROOT / "data/dataset_real_zarr/pick_cube_1cam/dataset.zarr.zip",
        "frequency_hz": 20.0,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data/outputs/2026.08.26/cross_eval_mse/domain_gap_analysis",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-samples-per-domain", type=int, default=1200)
    parser.add_argument("--action-windows-per-episode", type=int, default=64)
    return parser.parse_args()


def native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def scalar_summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p05": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "p50": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
    }


def choose_episode_balanced_indices(
    starts: np.ndarray,
    ends: np.ndarray,
    total: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    per_episode = max(1, math.ceil(total / len(ends)))
    frame_ids: list[int] = []
    episode_ids: list[int] = []
    for episode_id, (start, end) in enumerate(zip(starts, ends)):
        count = min(per_episode, int(end - start))
        selected = rng.choice(np.arange(start, end), size=count, replace=False)
        frame_ids.extend(int(x) for x in selected)
        episode_ids.extend([episode_id] * count)
    if len(frame_ids) > total:
        keep = rng.choice(len(frame_ids), size=total, replace=False)
        frame_ids = [frame_ids[i] for i in keep]
        episode_ids = [episode_ids[i] for i in keep]
    order = np.argsort(frame_ids)
    return np.asarray(frame_ids)[order], np.asarray(episode_ids)[order]


def image_features(image: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    image_f = image.astype(np.float32) / 255.0
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    edges = cv2.Canny(gray, 80, 160)

    rgb_mean = image_f.mean(axis=(0, 1))
    rgb_std = image_f.std(axis=(0, 1))
    intensity = image_f.mean(axis=-1)
    saturation = hsv[..., 1] / 255.0
    value = hsv[..., 2] / 255.0

    # Coarse masks are intentionally simple and fixed across both domains.
    # They measure framing/appearance, not semantic segmentation quality.
    red, green, blue = [image[..., i] for i in range(3)]
    orange = (red > 150) & (green > 45) & (green < 185) & (blue < 150) & (red > green * 1.25)
    dark = value < 0.25
    white = (value > 0.88) & (saturation < 0.18)

    summary = {
        "red_mean": float(rgb_mean[0]),
        "green_mean": float(rgb_mean[1]),
        "blue_mean": float(rgb_mean[2]),
        "red_std": float(rgb_std[0]),
        "green_std": float(rgb_std[1]),
        "blue_std": float(rgb_std[2]),
        "intensity_mean": float(intensity.mean()),
        "intensity_std": float(intensity.std()),
        "saturation_mean": float(saturation.mean()),
        "value_mean": float(value.mean()),
        "white_fraction": float(white.mean()),
        "dark_fraction": float(dark.mean()),
        "orange_fraction": float(orange.mean()),
        "edge_fraction": float((edges > 0).mean()),
        "laplacian_variance": float(laplacian.var()),
    }

    # A low-resolution spatial descriptor retains camera framing while keeping
    # the domain-classification diagnostic inexpensive and deterministic.
    coarse = cv2.resize(image_f, (12, 12), interpolation=cv2.INTER_AREA).reshape(-1)
    hist_parts = []
    for channel in range(3):
        hist, _ = np.histogram(image_f[..., channel], bins=16, range=(0.0, 1.0), density=True)
        hist_parts.append(hist.astype(np.float32))
    descriptor = np.concatenate([coarse, *hist_parts], axis=0)
    return descriptor, summary


def episode_split(episode_ids: np.ndarray, seed: int, train_fraction: float = 0.8) -> np.ndarray:
    unique = np.unique(episode_ids)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique)
    n_train = min(max(1, round(len(unique) * train_fraction)), len(unique) - 1)
    return np.isin(episode_ids, shuffled[:n_train])


def auc_score(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(bool)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    ranks = rankdata(scores)
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def pca_lda_domain_test(
    features: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    max_components: int = 24,
) -> dict[str, float]:
    x_train = features[train_mask].astype(np.float64)
    y_train = labels[train_mask].astype(np.int64)
    x_test = features[~train_mask].astype(np.float64)
    y_test = labels[~train_mask].astype(np.int64)

    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std < 1e-7] = 1.0
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    _, _, vt = np.linalg.svd(x_train - x_train.mean(axis=0), full_matrices=False)
    n_components = min(max_components, vt.shape[0], vt.shape[1])
    projection = vt[:n_components].T
    x_train = x_train @ projection
    x_test = x_test @ projection

    mu0 = x_train[y_train == 0].mean(axis=0)
    mu1 = x_train[y_train == 1].mean(axis=0)
    centered = np.concatenate([x_train[y_train == 0] - mu0, x_train[y_train == 1] - mu1], axis=0)
    covariance = centered.T @ centered / max(1, centered.shape[0] - 2)
    regularizer = max(float(np.trace(covariance) / covariance.shape[0]) * 0.1, 1e-5)
    covariance += np.eye(covariance.shape[0]) * regularizer
    direction = np.linalg.solve(covariance, mu1 - mu0)
    threshold = 0.5 * float((mu1 + mu0) @ direction)
    scores = x_test @ direction - threshold
    predictions = scores >= 0
    acc0 = float(np.mean(predictions[y_test == 0] == 0))
    acc1 = float(np.mean(predictions[y_test == 1] == 1))
    auc = auc_score(y_test, scores)
    if auc < 0.5:
        auc = 1.0 - auc
    return {
        "balanced_accuracy": (acc0 + acc1) / 2,
        "auc": auc,
        "test_samples": int(len(y_test)),
        "pca_components": int(n_components),
    }


def build_relative_action_windows(
    pos: np.ndarray,
    rotvec: np.ndarray,
    gripper: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    frequency_hz: float,
    max_per_episode: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    horizon = 16
    stride = 3
    # Robot pose is latency-aligned to the camera in SequenceSampler.
    robot_latency_steps = (0.125 - 0.0001) * frequency_hz
    all_actions: list[np.ndarray] = []
    all_classifier_features: list[np.ndarray] = []
    all_episode_ids: list[int] = []

    for episode_id, (start, end) in enumerate(zip(starts, ends)):
        valid = np.arange(start, end - (horizon - 1) * stride)
        if len(valid) > max_per_episode:
            valid = np.sort(rng.choice(valid, size=max_per_episode, replace=False))
        local_times = np.arange(start, end, dtype=np.float64)
        episode_rot = Rotation.from_rotvec(rotvec[start:end])
        slerp = Slerp(local_times, episode_rot)
        base_times = np.clip(valid.astype(np.float64) + robot_latency_steps, start, end - 1)
        base_pos = np.stack([
            np.interp(base_times, local_times, pos[start:end, axis]) for axis in range(3)
        ], axis=-1)
        base_rot = slerp(base_times)

        future_idx = valid[:, None] + stride * np.arange(horizon)[None, :]
        future_pos = pos[future_idx]
        future_rot = Rotation.from_rotvec(rotvec[future_idx.reshape(-1)]).as_matrix().reshape(
            len(valid), horizon, 3, 3
        )
        base_matrix = base_rot.as_matrix()
        local_pos = np.einsum("bij,bhj->bhi", np.swapaxes(base_matrix, 1, 2), future_pos - base_pos[:, None])
        local_rot = np.einsum("bij,bhjk->bhik", np.swapaxes(base_matrix, 1, 2), future_rot)
        rot6d = mat_to_rot6d(local_rot)
        action = np.concatenate([local_pos, rot6d, gripper[future_idx, None]], axis=-1).astype(np.float32)

        relative_rotvec = Rotation.from_matrix(local_rot.reshape(-1, 3, 3)).as_rotvec().reshape(len(valid), horizon, 3)
        classifier = np.concatenate(
            [local_pos.reshape(len(valid), -1), relative_rotvec.reshape(len(valid), -1), gripper[future_idx]],
            axis=-1,
        ).astype(np.float32)
        all_actions.append(action)
        all_classifier_features.append(classifier)
        all_episode_ids.extend([episode_id] * len(valid))

    return (
        np.concatenate(all_actions, axis=0),
        np.concatenate(all_classifier_features, axis=0),
        np.asarray(all_episode_ids),
    )


def analyze_domain(domain: str, args: argparse.Namespace) -> dict[str, Any]:
    spec = DATASETS[domain]
    rng = np.random.default_rng(args.seed + (0 if domain == "umi" else 1000))
    with zarr.ZipStore(str(spec["path"]), mode="r") as store:
        root = zarr.open_group(store, mode="r")
        ends = root["meta/episode_ends"][:]
        starts = np.r_[0, ends[:-1]]
        pos = root["data/robot0_eef_pos"][:]
        rotvec = root["data/robot0_eef_rot_axis_angle"][:]
        gripper = root["data/robot0_gripper_width"][:, 0]
        image_ids, image_episode_ids = choose_episode_balanced_indices(
            starts, ends, args.image_samples_per_domain, rng
        )
        descriptors = []
        image_rows = []
        images = root["data/camera0_rgb"]
        for frame_id, episode_id in zip(image_ids, image_episode_ids):
            descriptor, row = image_features(images[int(frame_id)])
            descriptors.append(descriptor)
            row.update({"domain": domain, "frame_id": int(frame_id), "episode_id": int(episode_id)})
            image_rows.append(row)

    frequency = float(spec["frequency_hz"])
    lengths = ends - starts
    episode_rows = []
    phase_rows = []
    phase_accumulator = {
        phase: {"speed": [], "angular_speed": [], "gripper": [], "start_displacement": []}
        for phase in range(10)
    }
    for episode_id, (start, end) in enumerate(zip(starts, ends)):
        ep_pos = pos[start:end]
        ep_rot = Rotation.from_rotvec(rotvec[start:end])
        ep_grip = gripper[start:end]
        speed = np.r_[0.0, np.linalg.norm(np.diff(ep_pos, axis=0), axis=-1) * frequency]
        angular_speed = np.r_[0.0, (ep_rot[:-1].inv() * ep_rot[1:]).magnitude() * frequency]
        path_length = float(np.linalg.norm(np.diff(ep_pos, axis=0), axis=-1).sum())
        net_displacement = float(np.linalg.norm(ep_pos[-1] - ep_pos[0]))
        closed = ep_grip < 0.065
        close_idx = np.flatnonzero(closed)
        close_phase = float(close_idx[0] / max(1, len(ep_grip) - 1)) if len(close_idx) else float("nan")
        reopen_idx = (
            np.flatnonzero((np.arange(len(ep_grip)) > close_idx[0]) & (ep_grip > 0.075))
            if len(close_idx)
            else np.asarray([])
        )
        reopen_phase = float(reopen_idx[0] / max(1, len(ep_grip) - 1)) if len(reopen_idx) else float("nan")
        episode_rows.append({
            "domain": domain,
            "episode_id": episode_id,
            "frames": int(lengths[episode_id]),
            "duration_s": float(lengths[episode_id] / frequency),
            "path_length_m": path_length,
            "net_displacement_m": net_displacement,
            "mean_speed_m_s": float(speed.mean()),
            "median_speed_m_s": float(np.median(speed)),
            "mean_angular_speed_rad_s": float(angular_speed.mean()),
            "closed_fraction": float(closed.mean()),
            "first_close_phase": close_phase,
            "reopen_phase": reopen_phase,
        })
        phases = np.minimum((np.arange(len(ep_pos)) / len(ep_pos) * 10).astype(int), 9)
        start_displacement = np.linalg.norm(ep_pos - ep_pos[0], axis=-1)
        for phase in range(10):
            mask = phases == phase
            phase_accumulator[phase]["speed"].extend(speed[mask])
            phase_accumulator[phase]["angular_speed"].extend(angular_speed[mask])
            phase_accumulator[phase]["gripper"].extend(ep_grip[mask])
            phase_accumulator[phase]["start_displacement"].extend(start_displacement[mask])

    for phase in range(10):
        values = phase_accumulator[phase]
        phase_rows.append({
            "domain": domain,
            "phase_bin": phase,
            "phase_start": phase / 10,
            "phase_end": (phase + 1) / 10,
            "mean_speed_m_s": float(np.mean(values["speed"])),
            "mean_angular_speed_rad_s": float(np.mean(values["angular_speed"])),
            "mean_gripper_width_m": float(np.mean(values["gripper"])),
            "mean_displacement_from_start_m": float(np.mean(values["start_displacement"])),
        })

    actions, action_classifier, action_episode_ids = build_relative_action_windows(
        pos,
        rotvec,
        gripper,
        starts,
        ends,
        frequency,
        args.action_windows_per_episode,
        rng,
    )
    episode_df = pd.DataFrame(episode_rows)
    return {
        "metadata": {
            "frames": int(len(pos)),
            "episodes": int(len(ends)),
            "frequency_hz": frequency,
        },
        "trajectory_summary": {
            key: scalar_summary(episode_df[key].dropna().to_numpy())
            for key in [
                "frames", "duration_s", "path_length_m", "net_displacement_m",
                "mean_speed_m_s", "median_speed_m_s", "mean_angular_speed_rad_s",
                "closed_fraction", "first_close_phase", "reopen_phase",
            ]
        },
        "position_absolute": {
            "mean_m": pos.mean(axis=0),
            "std_m": pos.std(axis=0),
            "start_mean_m": np.stack([pos[start] for start in starts]).mean(axis=0),
            "start_std_m": np.stack([pos[start] for start in starts]).std(axis=0),
            "end_mean_m": np.stack([pos[end - 1] for end in ends]).mean(axis=0),
            "end_std_m": np.stack([pos[end - 1] for end in ends]).std(axis=0),
        },
        "action_summary": {
            "windows": int(len(actions)),
            "position_mean_m": actions[..., :3].reshape(-1, 3).mean(axis=0),
            "position_std_m": actions[..., :3].reshape(-1, 3).std(axis=0),
            "position_min_m": actions[..., :3].reshape(-1, 3).min(axis=0),
            "position_max_m": actions[..., :3].reshape(-1, 3).max(axis=0),
            "position_norm_mean_m": float(np.linalg.norm(actions[..., :3], axis=-1).mean()),
            "position_norm_p95_m": float(np.percentile(np.linalg.norm(actions[..., :3], axis=-1), 95)),
            "rotation_angle_mean_rad": float(
                np.linalg.norm(action_classifier[:, 48:96].reshape(-1, 3), axis=-1).mean()
            ),
            "rotation_angle_p95_rad": float(
                np.percentile(np.linalg.norm(action_classifier[:, 48:96].reshape(-1, 3), axis=-1), 95)
            ),
            "gripper_mean_m": float(actions[..., 9].mean()),
            "gripper_std_m": float(actions[..., 9].std()),
        },
        "image_rows": image_rows,
        "image_descriptors": np.stack(descriptors),
        "image_episode_ids": image_episode_ids,
        "episode_rows": episode_rows,
        "phase_rows": phase_rows,
        "actions": actions,
        "action_classifier": action_classifier,
        "action_episode_ids": action_episode_ids,
    }


def load_action_normalizers() -> dict[str, dict[str, np.ndarray]]:
    result = {}
    pattern = PROJECT_ROOT / "data/outputs/2026.08.26/*pick_cube*/normalizer.pkl"
    for path_text in glob.glob(str(pattern)):
        path = Path(path_text)
        name = path.parent.name
        domain = "umi" if "_umi_" in name else "real"
        if domain in result:
            continue
        with path.open("rb") as file:
            normalizer = pickle.load(file)
        state = normalizer.state_dict()
        result[domain] = {
            "min": state["params_dict.action.input_stats.min"].cpu().numpy(),
            "max": state["params_dict.action.input_stats.max"].cpu().numpy(),
            "mean": state["params_dict.action.input_stats.mean"].cpu().numpy(),
            "std": state["params_dict.action.input_stats.std"].cpu().numpy(),
            "scale": state["params_dict.action.scale"].cpu().numpy(),
            "offset": state["params_dict.action.offset"].cpu().numpy(),
        }
    return result


def support_mismatch(
    source_domain: str,
    target_domain: str,
    target_actions: np.ndarray,
    normalizers: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    source = normalizers[source_domain]
    target_flat = target_actions.reshape(-1, 10)
    below = target_flat < source["min"]
    above = target_flat > source["max"]
    outside = below | above
    clipped = np.clip(target_flat, source["min"], source["max"])
    lower_bound = np.square(target_flat - clipped)
    position_outside_per_axis = outside[:, :3].mean(axis=0)
    return {
        "source_domain": source_domain,
        "target_domain": target_domain,
        "position_outside_fraction_per_axis": position_outside_per_axis,
        "position_outside_element_fraction": float(outside[:, :3].mean()),
        "action_steps_with_any_position_outside_fraction": float(outside[:, :3].any(axis=-1).mean()),
        "windows_with_any_position_outside_fraction": float(outside[:, :3].reshape(-1, 16, 3).any(axis=(1, 2)).mean()),
        "position_clip_lower_bound_mse": float(lower_bound[:, :3].mean()),
        "position_clip_lower_bound_3d_rmse_m": float(np.sqrt(3 * lower_bound[:, :3].mean())),
        "gripper_outside_fraction": float(outside[:, 9].mean()),
        "source_position_min_m": source["min"][:3],
        "source_position_max_m": source["max"][:3],
        "target_position_min_m": target_flat[:, :3].min(axis=0),
        "target_position_max_m": target_flat[:, :3].max(axis=0),
    }


def parse_training_logs() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries = []
    curves = []
    pattern = PROJECT_ROOT / "data/outputs/2026.08.26/*pick_cube*/logs.json.txt"
    for path_text in sorted(glob.glob(str(pattern))):
        path = Path(path_text)
        name = path.parent.name
        domain = "umi" if "_umi_" in name else "real"
        state = "no_state" if name.endswith("_no_state") else "with_state"
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        validation = [x for x in records if "val/action_mse_error" in x]
        lrs = np.asarray([x["train/lr"] for x in records])
        peak_lr_index = int(np.argmax(lrs))
        best = min(validation, key=lambda x: x["val/action_mse_error"])
        final = validation[-1]
        epoch_180 = min(validation, key=lambda x: abs(x["train/epoch"] - 180))
        summaries.append({
            "domain": domain,
            "state": state,
            "logged_batches": len(records),
            "approx_optimizer_steps": int((len(records) + 1) // 2),
            "peak_lr": float(lrs[peak_lr_index]),
            "peak_lr_global_step": int(records[peak_lr_index]["global_step"]),
            "peak_lr_epoch": int(records[peak_lr_index]["train/epoch"]),
            "best_val_epoch": int(best["train/epoch"]),
            "best_val_mse": float(best["val/action_mse_error"]),
            "final_val_mse": float(final["val/action_mse_error"]),
            "final_position_mse": float(final["val_action_mse_error_pos"]),
            "final_rotation_mse": float(final["val_action_mse_error_rot"]),
            "final_gripper_mse": float(final["val_action_mse_error_width"]),
            "final_over_best": float(final["val/action_mse_error"] / best["val/action_mse_error"]),
            "val_improvement_epoch180_to_final_fraction": float(
                1 - final["val/action_mse_error"] / epoch_180["val/action_mse_error"]
            ),
        })
        for row in validation:
            curves.append({
                "domain": domain,
                "state": state,
                "epoch": int(row["train/epoch"]),
                "global_step": int(row["global_step"]),
                "learning_rate": float(row["train/lr"]),
                "overall_mse": float(row["val/action_mse_error"]),
                "position_mse": float(row["val_action_mse_error_pos"]),
                "rotation_mse": float(row["val_action_mse_error_rot"]),
                "gripper_mse": float(row["val_action_mse_error_width"]),
            })
    return summaries, curves


def time_alignment_summary() -> dict[str, Any]:
    real_camera_age: dict[str, list[np.ndarray]] = {"usb": [], "top": []}
    real_root = PROJECT_ROOT / "data/real_data_lerobot_preprocess/pick_cube"
    for dataset in sorted(path for path in real_root.iterdir() if path.is_dir()):
        parquet_paths = sorted((dataset / "data").glob("chunk-*/*.parquet"))
        table = pq.read_table([str(path) for path in parquet_paths])
        for camera in ("usb", "top"):
            column = table[f"observation.camera_age.{camera}"].combine_chunks()
            values = np.asarray(column.to_numpy(), dtype=np.float64).reshape(-1)
            real_camera_age[camera].append(values)

    umi_usb_mean_ms = []
    umi_usb_max_ms = []
    umi_root = PROJECT_ROOT / "data/umi_data_lerobot_preprocess/pick_cube"
    for receipt in sorted(umi_root.glob("*/meta/preprocess.json")):
        payload = json.loads(receipt.read_text(encoding="utf-8"))
        synchronization = payload["synchronization"]
        umi_usb_mean_ms.append(float(synchronization["usb_error_ms"]["mean"]))
        umi_usb_max_ms.append(float(synchronization["usb_error_ms"]["max"]))

    return {
        "training_config_camera_obs_latency_s": 0.125,
        "real_recorded_camera_age_s": {
            camera: scalar_summary(np.concatenate(values))
            for camera, values in real_camera_age.items()
        },
        "umi_usb_to_clock_alignment_error_ms_across_episodes": {
            "episode_mean_error": scalar_summary(np.asarray(umi_usb_mean_ms)),
            "episode_max_error": scalar_summary(np.asarray(umi_usb_max_ms)),
        },
        "interpretation": (
            "Real per-frame camera age is discarded by the Zarr converter while both domains use "
            "a fixed 125 ms observation-latency setting. UMI images/trajectory are explicitly "
            "synchronized before conversion."
        ),
    }


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    domains = {domain: analyze_domain(domain, args) for domain in ("umi", "real")}

    image_features_all = np.concatenate([domains[d]["image_descriptors"] for d in ("umi", "real")])
    image_labels = np.concatenate([
        np.full(len(domains[d]["image_descriptors"]), i, dtype=np.int64)
        for i, d in enumerate(("umi", "real"))
    ])
    image_train_masks = []
    for i, domain in enumerate(("umi", "real")):
        image_train_masks.append(episode_split(domains[domain]["image_episode_ids"], args.seed + i))
    image_domain_test = pca_lda_domain_test(
        image_features_all, image_labels, np.concatenate(image_train_masks), max_components=24
    )

    action_features_all = np.concatenate([domains[d]["action_classifier"] for d in ("umi", "real")])
    action_labels = np.concatenate([
        np.full(len(domains[d]["action_classifier"]), i, dtype=np.int64)
        for i, d in enumerate(("umi", "real"))
    ])
    action_train_masks = []
    for i, domain in enumerate(("umi", "real")):
        action_train_masks.append(episode_split(domains[domain]["action_episode_ids"], args.seed + 10 + i))
    action_domain_test = pca_lda_domain_test(
        action_features_all, action_labels, np.concatenate(action_train_masks), max_components=24
    )

    image_rows = [row for domain in domains.values() for row in domain["image_rows"]]
    episode_rows = [row for domain in domains.values() for row in domain["episode_rows"]]
    phase_rows = [row for domain in domains.values() for row in domain["phase_rows"]]
    image_df = pd.DataFrame(image_rows)
    episode_df = pd.DataFrame(episode_rows)

    image_comparison = {}
    for column in [
        "intensity_mean", "intensity_std", "saturation_mean", "white_fraction",
        "dark_fraction", "orange_fraction", "edge_fraction", "laplacian_variance",
    ]:
        umi = image_df.loc[image_df.domain == "umi", column].to_numpy()
        real = image_df.loc[image_df.domain == "real", column].to_numpy()
        pooled_std = math.sqrt((float(np.var(umi)) + float(np.var(real))) / 2)
        image_comparison[column] = {
            "umi": scalar_summary(umi),
            "real": scalar_summary(real),
            "wasserstein": float(wasserstein_distance(umi, real)),
            "cohen_d": float((np.mean(real) - np.mean(umi)) / max(pooled_std, 1e-12)),
        }

    trajectory_comparison = {}
    for column in [
        "duration_s", "path_length_m", "net_displacement_m", "mean_speed_m_s",
        "median_speed_m_s", "mean_angular_speed_rad_s", "closed_fraction",
        "first_close_phase", "reopen_phase",
    ]:
        umi = episode_df.loc[episode_df.domain == "umi", column].dropna().to_numpy()
        real = episode_df.loc[episode_df.domain == "real", column].dropna().to_numpy()
        pooled_std = math.sqrt((float(np.var(umi)) + float(np.var(real))) / 2)
        trajectory_comparison[column] = {
            "umi": scalar_summary(umi),
            "real": scalar_summary(real),
            "wasserstein": float(wasserstein_distance(umi, real)),
            "cohen_d": float((np.mean(real) - np.mean(umi)) / max(pooled_std, 1e-12)),
        }

    normalizers = load_action_normalizers()
    support = [
        support_mismatch("real", "umi", domains["umi"]["actions"], normalizers),
        support_mismatch("umi", "real", domains["real"]["actions"], normalizers),
    ]
    training_summaries, training_curves = parse_training_logs()
    timing = time_alignment_summary()

    report_domains = {}
    for domain, data in domains.items():
        report_domains[domain] = {
            key: value for key, value in data.items()
            if key not in {
                "image_rows", "image_descriptors", "image_episode_ids", "episode_rows",
                "phase_rows", "actions", "action_classifier", "action_episode_ids",
            }
        }
    report = {
        "metadata": {
            "seed": args.seed,
            "image_samples_per_domain": args.image_samples_per_domain,
            "action_windows_per_episode": args.action_windows_per_episode,
            "action_horizon": 16,
            "action_downsample_steps": 3,
            "note": "Action windows reproduce the dataset's latency-aligned relative-pose convention.",
        },
        "domains": report_domains,
        "image_domain_classifier": image_domain_test,
        "relative_action_domain_classifier": action_domain_test,
        "image_comparison": image_comparison,
        "trajectory_comparison": trajectory_comparison,
        "action_normalizers": normalizers,
        "cross_domain_action_support": support,
        "training_summaries": training_summaries,
        "time_alignment": timing,
    }

    (args.output_dir / "domain_gap_analysis.json").write_text(
        json.dumps(native(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    pd.DataFrame(image_rows).to_csv(args.output_dir / "image_frame_metrics.csv", index=False)
    pd.DataFrame(episode_rows).to_csv(args.output_dir / "episode_metrics.csv", index=False)
    pd.DataFrame(phase_rows).to_csv(args.output_dir / "trajectory_phase_profiles.csv", index=False)
    pd.DataFrame(training_curves).to_csv(args.output_dir / "training_curves.csv", index=False)
    pd.DataFrame(training_summaries).to_csv(args.output_dir / "training_summary.csv", index=False)
    pd.DataFrame([
        {
            **{k: v for k, v in row.items() if not isinstance(v, (list, np.ndarray))},
            **{
                f"{k}_{axis}": float(v[i])
                for k, v in row.items() if isinstance(v, (list, np.ndarray))
                for i, axis in enumerate("xyz"[: len(v)])
            },
        }
        for row in support
    ]).to_csv(args.output_dir / "action_support_mismatch.csv", index=False)

    print(json.dumps(native({
        "image_domain_classifier": image_domain_test,
        "relative_action_domain_classifier": action_domain_test,
        "cross_domain_action_support": support,
        "training_summaries": training_summaries,
    }), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
