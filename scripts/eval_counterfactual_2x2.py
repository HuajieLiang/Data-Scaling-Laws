#!/usr/bin/env python3
"""Paired 2x2 counterfactual evaluation for UMI/real cross-domain policies.

The two factors are evaluated without retraining:

* image: original target-domain images vs. a per-channel photometric mapping
  (histogram or mean/std) into the checkpoint's source image distribution;
* trajectory: original target-domain labels vs. phase-rate retiming into the
  checkpoint's source episode tempo.

Only the first eight predicted actions are scored.  This keeps the accelerated
retiming within the original 16-step future support and targets the near-term
commands most relevant to safe execution.  Every image pair uses identical
diffusion noise, making the image counterfactual comparison paired.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dill
import hydra
import numpy as np
import torch
import zarr
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation, Slerp
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs  # noqa: E402
from diffusion_policy.common.pytorch_util import dict_apply  # noqa: E402
from umi.common.pose_util import mat_to_rot6d  # noqa: E402


register_codecs()
OmegaConf.register_new_resolver("eval", eval, replace=True)


@dataclass(frozen=True)
class DomainSpec:
    name: str
    dataset: Path
    frequency_hz: float
    mean_episode_duration_s: float
    checkpoint: Path


DOMAINS = {
    "umi": DomainSpec(
        name="umi",
        dataset=PROJECT_ROOT / "data/dataset_umi_zarr/pick_cube_1cam/dataset.zarr.zip",
        frequency_hz=23.12959389342191,
        mean_episode_duration_s=8.66007083788852,
        checkpoint=PROJECT_ROOT / "data/outputs/2026.08.26/26-16.56.40_pick_cube_umi_1cam_no_state/checkpoints/latest.ckpt",
    ),
    "real": DomainSpec(
        name="real",
        dataset=PROJECT_ROOT / "data/dataset_real_zarr/pick_cube_1cam/dataset.zarr.zip",
        frequency_hz=20.0,
        mean_episode_duration_s=19.09796511627907,
        checkpoint=PROJECT_ROOT / "data/outputs/2026.08.26/26-16.59.45_pick_cube_real_1cam_no_state/checkpoints/latest.ckpt",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data/outputs/2026.08.26/cross_eval_mse/counterfactual_2x2",
    )
    parser.add_argument("--samples-per-direction", type=int, default=2048)
    parser.add_argument("--histogram-frames", type=int, default=4096)
    parser.add_argument(
        "--image-transform",
        choices=("histogram", "rgb_moments"),
        default="histogram",
        help="Photometric counterfactual applied to target-domain images.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-threads", type=int, default=16)
    parser.add_argument("--directions", nargs="+", choices=("real_to_umi", "umi_to_real"), default=("real_to_umi", "umi_to_real"))
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


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_payload(path: Path) -> dict[str, Any]:
    return torch.load(str(path), map_location="cpu", pickle_module=dill, mmap=True)


def build_policy(spec: DomainSpec, device: torch.device):
    payload = load_payload(spec.checkpoint)
    cfg = payload["cfg"]
    cfg.policy.obs_encoder.pretrained = False
    policy = hydra.utils.instantiate(cfg.policy)
    policy.load_state_dict(payload["state_dicts"]["ema_model"], strict=True)
    del payload
    gc.collect()
    policy.eval().requires_grad_(False)
    return policy.to(device), cfg


def build_validation_dataset(cfg, target: DomainSpec, val_ratio: float, seed: int):
    dataset_cfg = copy.deepcopy(cfg.task.dataset)
    relative_path = os.path.relpath(target.dataset, PROJECT_ROOT)
    dataset_cfg.dataset_path = relative_path
    dataset_cfg.val_ratio = val_ratio
    dataset_cfg.seed = seed
    dataset_cfg.dataset_idx = None
    dataset_cfg.use_ratio = 1.0
    # Latency expressions in the copied shape_meta resolve through these values.
    cfg.task.dataset_frequeny = target.frequency_hz
    dataset_cfg.shape_meta = cfg.task.shape_meta
    training_dataset = hydra.utils.instantiate(dataset_cfg)
    validation_dataset = training_dataset.get_validation_dataset()
    del training_dataset
    return validation_dataset


def phase_stratified_indices(dataset, count: int, seed: int) -> np.ndarray:
    buckets: dict[int, list[int]] = defaultdict(list)
    for sample_index, (current, start, end, _) in enumerate(dataset.sampler.indices):
        phase = min(9, int(10 * (current - start) / max(1, end - start)))
        buckets[phase].append(sample_index)
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    quota = int(math.ceil(count / 10))
    for phase in range(10):
        values = np.asarray(buckets[phase], dtype=np.int64)
        take = min(quota, len(values))
        selected.extend(rng.choice(values, size=take, replace=False).tolist())
    if len(selected) > count:
        selected = rng.choice(np.asarray(selected), size=count, replace=False).tolist()
    return np.asarray(sorted(selected), dtype=np.int64)


def interp_linear(array: np.ndarray, times: np.ndarray, start: int, end: int) -> np.ndarray:
    base_times = np.arange(start, end, dtype=np.float64)
    return np.stack([
        np.interp(times, base_times, array[start:end, axis]) for axis in range(array.shape[-1])
    ], axis=-1)


def build_retimed_actions(dataset, indices: np.ndarray, source: DomainSpec, target: DomainSpec) -> tuple[np.ndarray, dict[str, float]]:
    sampler = dataset.sampler
    raw = sampler.replay_buffer
    pos = raw["robot0_eef_pos"]
    rotvec = raw["robot0_eef_rot_axis_angle"]
    gripper = raw["robot0_gripper_width"]
    latency = float(sampler.key_latency_steps["robot0_eef_pos"])
    episode_cache: dict[tuple[int, int], tuple[np.ndarray, Slerp]] = {}
    actions = []
    strides = []
    desired_strides = []
    capped = 0

    for sample_index in indices:
        current, start, end, _ = sampler.indices[int(sample_index)]
        key = (start, end)
        if key not in episode_cache:
            episode_times = np.arange(start, end, dtype=np.float64)
            episode_cache[key] = (episode_times, Slerp(episode_times, Rotation.from_rotvec(rotvec[start:end])))
        episode_times, slerp = episode_cache[key]

        target_duration = (end - start) / target.frequency_hz
        # Match the phase increment represented by one source-domain action step.
        phase_increment = (3.0 / source.frequency_hz) / source.mean_episode_duration_s
        desired_stride = phase_increment * target_duration * target.frequency_hz
        # The original sampler guarantees 45 raw future frames.  Restrict the
        # first-8 counterfactual to the future support available for this exact
        # sample instead of padding with an artificial stationary endpoint.
        max_supported_stride = (end - 1 - current) / 7.0
        counterfactual_stride = min(desired_stride, max_supported_stride)
        desired_strides.append(desired_stride)
        capped += int(counterfactual_stride < desired_stride - 1e-9)
        strides.append(counterfactual_stride)
        future_times = current + counterfactual_stride * np.arange(8, dtype=np.float64)
        if future_times[-1] > end - 1 + 1e-6:
            raise RuntimeError(
                f"Retimed first-8 target exceeds episode: {future_times[-1]} > {end - 1}"
            )
        base_time = np.clip(current + latency, start, end - 1)
        base_pos = interp_linear(pos, np.asarray([base_time]), start, end)[0]
        base_rot = slerp(np.asarray([base_time]))[0]
        future_pos = interp_linear(pos, future_times, start, end)
        future_rot = slerp(future_times)
        future_grip = interp_linear(gripper, future_times, start, end)

        local_pos = base_rot.inv().apply(future_pos - base_pos)
        local_rot = (base_rot.inv() * future_rot).as_matrix()
        rot6d = mat_to_rot6d(local_rot)
        actions.append(np.concatenate([local_pos, rot6d, future_grip], axis=-1).astype(np.float32))

    strides = np.asarray(strides)
    desired_strides = np.asarray(desired_strides)
    return np.stack(actions), {
        "counterfactual_stride_frames_mean": float(strides.mean()),
        "counterfactual_stride_frames_p05": float(np.percentile(strides, 5)),
        "counterfactual_stride_frames_p95": float(np.percentile(strides, 95)),
        "desired_stride_frames_mean_before_support_cap": float(desired_strides.mean()),
        "support_capped_sample_fraction": capped / len(indices),
        "original_stride_frames": 3.0,
    }


class CounterfactualSubset(Dataset):
    def __init__(self, dataset, indices: np.ndarray, retimed_actions: np.ndarray):
        self.dataset = dataset
        self.indices = indices
        self.retimed_actions = retimed_actions

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        result = self.dataset[int(self.indices[index])]
        result["retimed_action"] = torch.from_numpy(self.retimed_actions[index])
        result["selection_index"] = torch.tensor(index, dtype=torch.int64)
        return result


def dataset_rgb_histogram(spec: DomainSpec, frames: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    hist = np.zeros((3, 256), dtype=np.float64)
    with zarr.ZipStore(str(spec.dataset), mode="r") as store:
        root = zarr.open_group(store, mode="r")
        images = root["data/camera0_rgb"]
        selected = np.sort(rng.choice(len(images), size=min(frames, len(images)), replace=False))
        for index in selected:
            image = images[int(index)]
            for channel in range(3):
                hist[channel] += np.bincount(image[..., channel].reshape(-1), minlength=256)
    hist /= hist.sum(axis=1, keepdims=True)
    return hist


def histogram_lut(source_hist: np.ndarray, target_hist: np.ndarray) -> np.ndarray:
    source_cdf = np.cumsum(source_hist, axis=1)
    target_cdf = np.cumsum(target_hist, axis=1)
    lut = np.empty((3, 256), dtype=np.uint8)
    values = np.arange(256)
    for channel in range(3):
        lut[channel] = np.interp(target_cdf[channel], source_cdf[channel], values).round().clip(0, 255)
    return lut


def mapped_histogram(target_hist: np.ndarray, lut: np.ndarray) -> np.ndarray:
    result = np.zeros_like(target_hist)
    for channel in range(3):
        np.add.at(result[channel], lut[channel], target_hist[channel])
    return result


def cdf_l1(first: np.ndarray, second: np.ndarray) -> list[float]:
    return np.mean(np.abs(np.cumsum(first, axis=1) - np.cumsum(second, axis=1)), axis=1).tolist()


def apply_lut(obs: dict[str, torch.Tensor], lut: torch.Tensor) -> dict[str, torch.Tensor]:
    result = dict(obs)
    image = obs["camera0_rgb"]
    indices = torch.clamp(torch.round(image * 255.0), 0, 255).to(torch.long)
    channels = []
    for channel in range(3):
        channels.append(lut[channel][indices[:, :, channel]])
    mapped = torch.stack(channels, dim=2).to(image.dtype) / 255.0
    result["camera0_rgb"] = mapped
    return result


def rgb_moments(histogram: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.arange(256, dtype=np.float64)[None] / 255.0
    mean = np.sum(histogram * values, axis=1)
    variance = np.sum(histogram * np.square(values - mean[:, None]), axis=1)
    return mean, np.sqrt(np.maximum(variance, 1e-12))


def apply_rgb_moments(
    obs: dict[str, torch.Tensor],
    source_mean: torch.Tensor,
    source_std: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> dict[str, torch.Tensor]:
    result = dict(obs)
    image = obs["camera0_rgb"]
    broadcast_shape = [1] * image.ndim
    broadcast_shape[2] = 3
    source_mean = source_mean.reshape(broadcast_shape)
    source_std = source_std.reshape(broadcast_shape)
    target_mean = target_mean.reshape(broadcast_shape)
    target_std = target_std.reshape(broadcast_shape)
    result["camera0_rgb"] = torch.clamp(
        (image - target_mean) / target_std * source_std + source_mean,
        0.0,
        1.0,
    )
    return result


class MetricAccumulator:
    def __init__(self) -> None:
        self.squared = {key: 0.0 for key in ("overall", "position", "rotation_6d", "gripper_width")}
        self.count = {key: 0 for key in self.squared}
        self.position_vector_abs_sum = 0.0
        self.position_vectors = 0
        self.predicted_norm_sum = 0.0
        self.target_norm_sum = 0.0
        self.cosine_sum = 0.0
        self.cosine_count = 0
        self.samples = 0

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        prediction = prediction.float()
        target = target.float()
        diff = prediction - target
        pieces = {
            "overall": diff,
            "position": diff[..., :3],
            "rotation_6d": diff[..., 3:9],
            "gripper_width": diff[..., 9:10],
        }
        for key, value in pieces.items():
            self.squared[key] += value.square().sum(dtype=torch.float64).item()
            self.count[key] += value.numel()
        vector_error = torch.linalg.vector_norm(diff[..., :3], dim=-1)
        predicted_norm = torch.linalg.vector_norm(prediction[..., :3], dim=-1)
        target_norm = torch.linalg.vector_norm(target[..., :3], dim=-1)
        self.position_vector_abs_sum += vector_error.sum(dtype=torch.float64).item()
        self.position_vectors += vector_error.numel()
        self.predicted_norm_sum += predicted_norm.sum(dtype=torch.float64).item()
        self.target_norm_sum += target_norm.sum(dtype=torch.float64).item()
        valid = (predicted_norm > 1e-5) & (target_norm > 1e-5)
        if valid.any():
            cosine = torch.nn.functional.cosine_similarity(
                prediction[..., :3][valid], target[..., :3][valid], dim=-1
            )
            self.cosine_sum += cosine.sum(dtype=torch.float64).item()
            self.cosine_count += cosine.numel()
        self.samples += prediction.shape[0]

    def result(self) -> dict[str, Any]:
        mse = {key: self.squared[key] / self.count[key] for key in self.squared}
        return {
            "mse": mse,
            "position_vector_mae_m": self.position_vector_abs_sum / self.position_vectors,
            "position_vector_rmse_m": math.sqrt(3 * mse["position"]),
            "predicted_position_norm_mean_m": self.predicted_norm_sum / self.position_vectors,
            "target_position_norm_mean_m": self.target_norm_sum / self.position_vectors,
            "position_direction_cosine_mean": self.cosine_sum / max(1, self.cosine_count),
            "samples": self.samples,
            "action_steps": self.position_vectors,
        }


def evaluate_direction(
    source: DomainSpec,
    target: DomainSpec,
    histograms: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> dict[str, Any]:
    device = torch.device(args.device)
    print(f"Loading {source.name} no-state policy", flush=True)
    policy, cfg = build_policy(source, device)
    print(f"Loading {target.name} validation dataset", flush=True)
    dataset = build_validation_dataset(cfg, target, args.val_ratio, args.seed)
    indices = phase_stratified_indices(dataset, min(args.samples_per_direction, len(dataset)), args.seed)
    retimed_actions, retiming_info = build_retimed_actions(dataset, indices, source, target)
    selected = CounterfactualSubset(dataset, indices, retimed_actions)
    loader = DataLoader(
        selected,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
    )

    if args.image_transform == "histogram":
        image_condition_name = "histogram_matched"
        lut_np = histogram_lut(histograms[source.name], histograms[target.name])
        lut = torch.from_numpy(lut_np.astype(np.int64)).to(device)
        mapped = mapped_histogram(histograms[target.name], lut_np)
        image_info = {
            "mapping": f"{target.name}_to_{source.name}_per_channel_histogram",
            "cdf_l1_before_per_rgb_channel": cdf_l1(histograms[target.name], histograms[source.name]),
            "cdf_l1_after_per_rgb_channel": cdf_l1(mapped, histograms[source.name]),
        }

        def transform_image(obs):
            return apply_lut(obs, lut)
    else:
        image_condition_name = "rgb_moment_matched"
        source_mean_np, source_std_np = rgb_moments(histograms[source.name])
        target_mean_np, target_std_np = rgb_moments(histograms[target.name])
        source_mean = torch.from_numpy(source_mean_np).to(device=device, dtype=torch.float32)
        source_std = torch.from_numpy(source_std_np).to(device=device, dtype=torch.float32)
        target_mean = torch.from_numpy(target_mean_np).to(device=device, dtype=torch.float32)
        target_std = torch.from_numpy(target_std_np).to(device=device, dtype=torch.float32)
        image_info = {
            "mapping": f"{target.name}_to_{source.name}_per_channel_mean_std",
            "source_mean_rgb": source_mean_np.tolist(),
            "source_std_rgb": source_std_np.tolist(),
            "target_mean_rgb": target_mean_np.tolist(),
            "target_std_rgb": target_std_np.tolist(),
        }

        def transform_image(obs):
            return apply_rgb_moments(obs, source_mean, source_std, target_mean, target_std)

    image_conditions = ("original", image_condition_name)

    accumulators = {
        (image_condition, trajectory_condition): MetricAccumulator()
        for image_condition in image_conditions
        for trajectory_condition in ("original", "phase_retimed")
    }
    started = time.monotonic()
    with torch.inference_mode():
        progress = tqdm(loader, desc=f"{source.name}->{target.name} 2x2", dynamic_ncols=True)
        for batch_index, batch in enumerate(progress):
            obs = dict_apply(batch["obs"], lambda value: value.to(device, non_blocking=True))
            original_target = batch["action"][:, :8].to(device, non_blocking=True)
            retimed_target = batch["retimed_action"].to(device, non_blocking=True)
            targets = {"original": original_target, "phase_retimed": retimed_target}

            for image_index, image_condition in enumerate(image_conditions):
                condition_obs = obs if image_condition == "original" else transform_image(obs)
                # Reset before both image variants so their DDIM initial noise is identical.
                seed_everything(args.seed + batch_index)
                prediction = policy.predict_action(condition_obs, None)["action_pred"][:, :8]
                for trajectory_condition, target_action in targets.items():
                    accumulators[image_condition, trajectory_condition].update(prediction, target_action)
                del prediction, condition_obs
            del batch, obs, original_target, retimed_target

    cells = []
    for image_condition in image_conditions:
        for trajectory_condition in ("original", "phase_retimed"):
            cells.append({
                "image_condition": image_condition,
                "trajectory_condition": trajectory_condition,
                **accumulators[image_condition, trajectory_condition].result(),
            })
    del policy, dataset, selected, loader
    gc.collect()
    return {
        "model_domain": source.name,
        "data_domain": target.name,
        "state": "no_state",
        "checkpoint": str(source.checkpoint.relative_to(PROJECT_ROOT)),
        "dataset": str(target.dataset.relative_to(PROJECT_ROOT)),
        "selected_samples": len(indices),
        "retiming": retiming_info,
        "image_counterfactual": image_info,
        "cells": cells,
        "elapsed_seconds": time.monotonic() - started,
    }


def add_effects(result: dict[str, Any]) -> dict[str, Any]:
    cells = {(x["image_condition"], x["trajectory_condition"]): x for x in result["cells"]}
    transformed_condition = next(x for x in {key[0] for key in cells} if x != "original")
    baseline = cells["original", "original"]
    image_only = cells[transformed_condition, "original"]
    trajectory_only = cells["original", "phase_retimed"]
    both = cells[transformed_condition, "phase_retimed"]

    def relative_reduction(candidate: dict[str, Any], metric_path: tuple[str, ...]) -> float:
        def get(row):
            value: Any = row
            for key in metric_path:
                value = value[key]
            return float(value)
        return 1.0 - get(candidate) / get(baseline)

    result["effects_relative_to_original_original"] = {
        "image_only_position_mse_reduction_fraction": relative_reduction(image_only, ("mse", "position")),
        "trajectory_only_position_mse_reduction_fraction": relative_reduction(trajectory_only, ("mse", "position")),
        "both_position_mse_reduction_fraction": relative_reduction(both, ("mse", "position")),
        "image_only_position_vector_mae_reduction_fraction": relative_reduction(image_only, ("position_vector_mae_m",)),
        "trajectory_only_position_vector_mae_reduction_fraction": relative_reduction(trajectory_only, ("position_vector_mae_m",)),
        "both_position_vector_mae_reduction_fraction": relative_reduction(both, ("position_vector_mae_m",)),
    }
    return result


def write_outputs(output_dir: Path, report: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "counterfactual_2x2.json").write_text(
        json.dumps(native(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    rows = []
    for result in report["results"]:
        for cell in result["cells"]:
            rows.append({
                "model_domain": result["model_domain"],
                "data_domain": result["data_domain"],
                "image_condition": cell["image_condition"],
                "trajectory_condition": cell["trajectory_condition"],
                "samples": cell["samples"],
                "mse_position": cell["mse"]["position"],
                "mse_rotation_6d": cell["mse"]["rotation_6d"],
                "mse_gripper_width": cell["mse"]["gripper_width"],
                "position_vector_mae_m": cell["position_vector_mae_m"],
                "position_vector_rmse_m": cell["position_vector_rmse_m"],
                "predicted_position_norm_mean_m": cell["predicted_position_norm_mean_m"],
                "target_position_norm_mean_m": cell["target_position_norm_mean_m"],
                "position_direction_cosine_mean": cell["position_direction_cosine_mean"],
            })
    fields = list(rows[0])
    with (output_dir / "counterfactual_2x2.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Cross-domain counterfactual 2x2 (first 8 actions)",
        "",
        "Paired DDIM noise; no-state EMA policies; held-out episodes.",
        "",
        "| Model → data | Image | Trajectory | Position MSE | 3D RMSE (mm) | Vector MAE (mm) | Pred norm (mm) | Target norm (mm) | Direction cosine |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model_domain']} → {row['data_domain']} | {row['image_condition']} | "
            f"{row['trajectory_condition']} | {row['mse_position']:.8g} | "
            f"{row['position_vector_rmse_m'] * 1000:.3f} | {row['position_vector_mae_m'] * 1000:.3f} | "
            f"{row['predicted_position_norm_mean_m'] * 1000:.3f} | "
            f"{row['target_position_norm_mean_m'] * 1000:.3f} | {row['position_direction_cosine_mean']:.4f} |"
        )
    (output_dir / "counterfactual_2x2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    os.chdir(PROJECT_ROOT)
    args.output_dir = args.output_dir.resolve()
    torch.set_num_threads(args.num_threads)
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    print("Computing dataset RGB histograms", flush=True)
    histograms = {
        domain: dataset_rgb_histogram(spec, args.histogram_frames, args.seed + index)
        for index, (domain, spec) in enumerate(DOMAINS.items())
    }
    results = []
    for direction in args.directions:
        source_name, target_name = direction.split("_to_")
        result = evaluate_direction(DOMAINS[source_name], DOMAINS[target_name], histograms, args)
        results.append(add_effects(result))
        write_outputs(args.output_dir, {
            "metadata": {
                "seed": args.seed,
                "val_ratio": args.val_ratio,
                "samples_per_direction": args.samples_per_direction,
                "histogram_frames": args.histogram_frames,
                "action_steps_scored": 8,
                "weights": "ema_model",
                "state": "no_state",
                "paired_diffusion_noise": True,
                "trajectory_counterfactual": "target trajectory sampled at source mean episode phase-rate",
                "image_counterfactual": (
                    "per-RGB-channel dataset histogram mapping target to source"
                    if args.image_transform == "histogram"
                    else "per-RGB-channel mean/std mapping target to source"
                ),
                "image_transform": args.image_transform,
            },
            "results": results,
        })

    report = {
        "metadata": {
            "seed": args.seed,
            "val_ratio": args.val_ratio,
            "samples_per_direction": args.samples_per_direction,
            "histogram_frames": args.histogram_frames,
            "action_steps_scored": 8,
            "weights": "ema_model",
            "state": "no_state",
            "paired_diffusion_noise": True,
            "trajectory_counterfactual": "target trajectory sampled at source mean episode phase-rate",
            "image_counterfactual": (
                "per-RGB-channel dataset histogram mapping target to source"
                if args.image_transform == "histogram"
                else "per-RGB-channel mean/std mapping target to source"
            ),
            "image_transform": args.image_transform,
        },
        "results": results,
    }
    write_outputs(args.output_dir, report)
    print(json.dumps(native(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
