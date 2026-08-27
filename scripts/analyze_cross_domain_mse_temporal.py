#!/usr/bin/env python3
"""Temporal and distribution analysis for UMI/real cross-domain action MSE."""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

import eval_cross_domain_mse as base


PROJECT_ROOT = Path(__file__).resolve().parents[1]
N_PHASE_BINS = 10
PERCENTILES = (0, 25, 50, 75, 90, 95, 99, 100)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=PROJECT_ROOT / "data/outputs/2026.08.26")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data/outputs/2026.08.26/cross_eval_mse/temporal",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def empty_group(size: int) -> dict[str, np.ndarray]:
    keys = ("overall", "without_gripper", "position", "rotation_6d", "gripper_width")
    return {f"{key}_sum": np.zeros(size, dtype=np.float64) for key in keys} | {
        f"{key}_count": np.zeros(size, dtype=np.int64) for key in keys
    }


def add_group(group: dict[str, np.ndarray], index: int, diff: torch.Tensor) -> None:
    # diff has arbitrary leading axes and a final 10D action axis.
    slices = {
        "overall": diff,
        "without_gripper": diff[..., :9],
        "position": diff[..., :3],
        "rotation_6d": diff[..., 3:9],
        "gripper_width": diff[..., 9:10],
    }
    for key, value in slices.items():
        group[f"{key}_sum"][index] += value.square().sum(dtype=torch.float64).item()
        group[f"{key}_count"][index] += value.numel()


def group_means(group: dict[str, np.ndarray], index: int) -> dict[str, float]:
    result = {}
    for key in ("overall", "without_gripper", "position", "rotation_6d", "gripper_width"):
        count = int(group[f"{key}_count"][index])
        result[key] = float(group[f"{key}_sum"][index] / count) if count else float("nan")
    return result


def prediction_origin_phase(dataset, dataset_index: int) -> tuple[float, bool, int, int]:
    current_idx, start_idx, end_idx, before_first_grasp = dataset.sampler.indices[dataset_index]
    horizon = int(dataset.key_horizon["action"])
    downsample = int(dataset.key_down_sample_steps["action"])
    last_valid_current = end_idx - ((horizon - 1) * downsample + 1)
    denominator = max(last_valid_current - start_idx, 1)
    phase = float(np.clip((current_idx - start_idx) / denominator, 0.0, 1.0))
    episode = int(np.searchsorted(dataset.replay_buffer.episode_ends[:], current_idx, side="right"))
    return phase, bool(before_first_grasp), episode, current_idx


def percentiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {f"p{p}": float(np.percentile(array, p)) for p in PERCENTILES}


def evaluate_temporal(policy, dataset, data_domain: str, args: argparse.Namespace, description: str) -> dict[str, Any]:
    base.seed_everything(args.seed)
    loader = base.make_loader(dataset, args)
    horizon_steps = int(dataset.key_horizon["action"])
    action_downsample = int(dataset.key_down_sample_steps["action"])
    seconds_per_step = action_downsample / base.DATASETS[data_domain].frequency

    by_horizon = empty_group(horizon_steps)
    by_episode_phase = empty_group(N_PHASE_BINS)
    by_gripper_phase = empty_group(2)
    episode_phase_sample_count = np.zeros(N_PHASE_BINS, dtype=np.int64)
    gripper_phase_sample_count = np.zeros(2, dtype=np.int64)
    sample_overall: list[float] = []
    sample_without_gripper: list[float] = []
    sample_phases: list[float] = []
    sample_episodes: list[int] = []
    sample_current_indices: list[int] = []
    samples = 0

    with torch.inference_mode():
        for batch in tqdm(loader, desc=description, dynamic_ncols=True):
            batch = base.dict_apply(batch, lambda x: x.to(args.device, non_blocking=True))
            ground_truth = batch["action"]
            with base.autocast_context(args):
                prediction = policy.predict_action(batch["obs"], None)["action_pred"]
            prediction = prediction.float().reshape(prediction.shape[0], prediction.shape[1], -1, 10)
            ground_truth = ground_truth.float().reshape(ground_truth.shape[0], ground_truth.shape[1], -1, 10)
            diff = prediction - ground_truth

            for step in range(horizon_steps):
                add_group(by_horizon, step, diff[:, step])

            overall_per_sample = diff.square().mean(dim=(1, 2, 3)).cpu().numpy()
            no_gripper_per_sample = diff[..., :9].square().mean(dim=(1, 2, 3)).cpu().numpy()
            batch_size = prediction.shape[0]
            for local_index in range(batch_size):
                dataset_index = samples + local_index
                phase, before_first_grasp, episode, current_idx = prediction_origin_phase(dataset, dataset_index)
                phase_bin = min(int(phase * N_PHASE_BINS), N_PHASE_BINS - 1)
                gripper_phase = 0 if before_first_grasp else 1
                add_group(by_episode_phase, phase_bin, diff[local_index])
                add_group(by_gripper_phase, gripper_phase, diff[local_index])
                episode_phase_sample_count[phase_bin] += 1
                gripper_phase_sample_count[gripper_phase] += 1
                sample_overall.append(float(overall_per_sample[local_index]))
                sample_without_gripper.append(float(no_gripper_per_sample[local_index]))
                sample_phases.append(phase)
                sample_episodes.append(episode)
                sample_current_indices.append(current_idx)
            samples += batch_size

            del batch, ground_truth, prediction, diff

    horizon_rows = [
        {
            "step": step,
            "time_seconds": step * seconds_per_step,
            "mse": group_means(by_horizon, step),
        }
        for step in range(horizon_steps)
    ]
    phase_rows = [
        {
            "bin": phase_bin,
            "start_fraction": phase_bin / N_PHASE_BINS,
            "end_fraction": (phase_bin + 1) / N_PHASE_BINS,
            "samples": int(episode_phase_sample_count[phase_bin]),
            "mse": group_means(by_episode_phase, phase_bin),
        }
        for phase_bin in range(N_PHASE_BINS)
    ]
    gripper_rows = [
        {
            "phase": label,
            "samples": int(gripper_phase_sample_count[index]),
            "mse": group_means(by_gripper_phase, index),
        }
        for index, label in enumerate(("before_first_gripper_close", "after_first_gripper_close"))
    ]

    no_gripper_array = np.asarray(sample_without_gripper)
    p90 = float(np.percentile(no_gripper_array, 90))
    tail_indices = np.flatnonzero(no_gripper_array >= p90)
    tail_phase_histogram, _ = np.histogram(
        np.asarray(sample_phases)[tail_indices], bins=np.linspace(0.0, 1.0, N_PHASE_BINS + 1)
    )
    highest_indices = np.argsort(no_gripper_array)[-20:][::-1]
    highest_samples = [
        {
            "rank": rank + 1,
            "dataset_index": int(index),
            "episode": int(sample_episodes[index]),
            "current_frame": int(sample_current_indices[index]),
            "episode_phase": float(sample_phases[index]),
            "mse_without_gripper": float(sample_without_gripper[index]),
            "mse_overall": float(sample_overall[index]),
        }
        for rank, index in enumerate(highest_indices)
    ]

    return {
        "samples_evaluated": samples,
        "seconds_per_action_step": seconds_per_step,
        "by_prediction_horizon": horizon_rows,
        "by_episode_phase": phase_rows,
        "by_gripper_phase": gripper_rows,
        "sample_mse_distribution": {
            "overall": percentiles(sample_overall),
            "without_gripper": percentiles(sample_without_gripper),
        },
        "top_10pct_without_gripper": {
            "threshold": p90,
            "phase_bin_counts": tail_phase_histogram.tolist(),
            "highest_20_samples": highest_samples,
        },
    }


def key(row: dict[str, Any]) -> tuple[str, str, str]:
    return row["model_domain"], row["state"], row["data_domain"]


def write_csvs(output_dir: Path, results: list[dict[str, Any]]) -> None:
    metric_names = ("overall", "without_gripper", "position", "rotation_6d", "gripper_width")

    with (output_dir / "mse_by_prediction_horizon.csv").open("w", newline="", encoding="utf-8") as file:
        fields = ["model_domain", "state", "data_domain", "step", "time_seconds"] + [f"mse_{x}" for x in metric_names]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for result in sorted(results, key=key):
            for item in result["by_prediction_horizon"]:
                writer.writerow({
                    "model_domain": result["model_domain"], "state": result["state"],
                    "data_domain": result["data_domain"], "step": item["step"],
                    "time_seconds": item["time_seconds"],
                    **{f"mse_{name}": item["mse"][name] for name in metric_names},
                })

    with (output_dir / "mse_by_episode_phase.csv").open("w", newline="", encoding="utf-8") as file:
        fields = ["model_domain", "state", "data_domain", "phase_start", "phase_end", "samples"] + [f"mse_{x}" for x in metric_names]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for result in sorted(results, key=key):
            for item in result["by_episode_phase"]:
                writer.writerow({
                    "model_domain": result["model_domain"], "state": result["state"],
                    "data_domain": result["data_domain"], "phase_start": item["start_fraction"],
                    "phase_end": item["end_fraction"], "samples": item["samples"],
                    **{f"mse_{name}": item["mse"][name] for name in metric_names},
                })

    with (output_dir / "mse_by_gripper_phase.csv").open("w", newline="", encoding="utf-8") as file:
        fields = ["model_domain", "state", "data_domain", "phase", "samples"] + [f"mse_{x}" for x in metric_names]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for result in sorted(results, key=key):
            for item in result["by_gripper_phase"]:
                writer.writerow({
                    "model_domain": result["model_domain"], "state": result["state"],
                    "data_domain": result["data_domain"], "phase": item["phase"], "samples": item["samples"],
                    **{f"mse_{name}": item["mse"][name] for name in metric_names},
                })

    with (output_dir / "sample_mse_percentiles.csv").open("w", newline="", encoding="utf-8") as file:
        percentile_names = tuple(f"p{p}" for p in PERCENTILES)
        fields = ["model_domain", "state", "data_domain", "dimensions"] + list(percentile_names)
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for result in sorted(results, key=key):
            for dimensions in ("overall", "without_gripper"):
                writer.writerow({
                    "model_domain": result["model_domain"], "state": result["state"],
                    "data_domain": result["data_domain"], "dimensions": dimensions,
                    **result["sample_mse_distribution"][dimensions],
                })


def write_outputs(output_dir: Path, metadata: dict[str, Any], results: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    results = sorted(results, key=key)
    (output_dir / "temporal_analysis.json").write_text(
        json.dumps({"metadata": metadata, "results": results}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csvs(output_dir, results)


def main() -> None:
    args = parse_args()
    os.chdir(PROJECT_ROOT)
    args.runs_dir = args.runs_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.precision == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA device does not support BF16")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    checkpoints = base.discover_checkpoints(args.runs_dir)
    expected = {(domain, state) for domain in ("umi", "real") for state in ("no_state", "with_state")}
    actual = {(spec.domain, spec.state) for spec in checkpoints}
    if not expected.issubset(actual):
        raise RuntimeError(f"Missing checkpoints: {sorted(expected - actual)}")
    checkpoints = [spec for spec in checkpoints if (spec.domain, spec.state) in expected]

    payload = base.load_payload(checkpoints[0].checkpoint)
    base_cfg = copy.deepcopy(payload["cfg"])
    del payload
    datasets = {}
    dataset_info = {}
    for domain in ("umi", "real"):
        print(f"Building {domain} validation dataset...", flush=True)
        datasets[domain], dataset_info[domain] = base.build_validation_dataset(
            base_cfg, base.DATASETS[domain], args.val_ratio, args.seed
        )
        print(json.dumps({domain: dataset_info[domain]}, ensure_ascii=False), flush=True)

    metadata = {
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "batch_size": args.batch_size,
        "precision": args.precision,
        "weights": "ema_model",
        "episode_phase_bins": N_PHASE_BINS,
        "dataset_info": dataset_info,
    }
    result_path = args.output_dir / "temporal_analysis.json"
    results: list[dict[str, Any]] = []
    if result_path.is_file() and not args.overwrite:
        old = json.loads(result_path.read_text(encoding="utf-8"))
        comparable = ("seed", "val_ratio", "batch_size", "precision", "weights", "episode_phase_bins")
        if any(old.get("metadata", {}).get(name) != metadata[name] for name in comparable):
            raise RuntimeError(f"Incompatible existing result: {result_path}; pass --overwrite")
        results = old.get("results", [])
    completed = {key(row) for row in results}

    for checkpoint in checkpoints:
        pending = [domain for domain in ("umi", "real") if (checkpoint.domain, checkpoint.state, domain) not in completed]
        if not pending:
            continue
        print(f"Loading {checkpoint.domain}/{checkpoint.state} EMA policy...", flush=True)
        policy, _ = base.build_policy(checkpoint, "ema_model", device)
        for domain in pending:
            description = f"temporal {checkpoint.domain}/{checkpoint.state} -> {domain}"
            metrics = evaluate_temporal(policy, datasets[domain], domain, args, description)
            row = {
                "model_domain": checkpoint.domain,
                "state": checkpoint.state,
                "data_domain": domain,
                "checkpoint": str(checkpoint.checkpoint.relative_to(PROJECT_ROOT)),
                **metrics,
            }
            results.append(row)
            completed.add(key(row))
            write_outputs(args.output_dir, metadata, results)
            summary = {
                "key": key(row),
                "horizon_no_gripper": [x["mse"]["without_gripper"] for x in row["by_prediction_horizon"]],
                "phase_no_gripper": [x["mse"]["without_gripper"] for x in row["by_episode_phase"]],
            }
            print(json.dumps(summary, ensure_ascii=False), flush=True)
        policy.to("cpu")
        del policy
        gc.collect()
        torch.cuda.empty_cache()

    write_outputs(args.output_dir, metadata, results)
    print(f"Temporal analysis written to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
