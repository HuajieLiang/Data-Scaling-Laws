#!/usr/bin/env python3
"""Convert cross-evaluation position MSE into practical distance errors."""

from __future__ import annotations

import copy
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

import eval_cross_domain_mse as base

PROJECT_ROOT = Path(__file__).resolve().parents[1]
N_EXECUTED_STEPS = 8


def position_metrics(diff: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """Return axis and 3D-vector metrics in metres for B,T,R,3 tensors."""
    error_norm = diff.norm(dim=-1)
    target_norm = target.norm(dim=-1)
    return {
        "axis_mae_m": diff.abs().mean().item(),
        "vector_mae_m": error_norm.mean().item(),
        "vector_rmse_m": diff.square().mean().sqrt().mul(3.0**0.5).item(),
        "target_vector_mean_m": target_norm.mean().item(),
        "target_vector_median_m": target_norm.median().item(),
        "target_vector_p90_m": torch.quantile(target_norm.flatten(), 0.9).item(),
    }


def evaluate(policy, dataset, data_domain: str, args: Any, description: str) -> dict[str, Any]:
    base.seed_everything(args.seed)
    loader = base.make_loader(dataset, args)
    horizon = int(dataset.key_horizon["action"])
    horizon_slices = {
        "executed_first_8": slice(0, min(N_EXECUTED_STEPS, horizon)),
        "full_16": slice(0, horizon),
    }
    sums: dict[str, dict[str, float]] = {}
    counts: dict[str, int] = {}
    sample_error_norms: dict[str, list[float]] = {name: [] for name in horizon_slices}
    sample_target_norms: dict[str, list[float]] = {name: [] for name in horizon_slices}
    by_step: list[dict[str, float]] = []
    for step in range(horizon):
        by_step.append({
            "step": step,
            "time_seconds": step * int(dataset.key_down_sample_steps["action"]) / base.DATASETS[data_domain].frequency,
            "error_sq_sum_m2": 0.0,
            "error_vector_count": 0,
            "error_abs_sum_m": 0.0,
            "target_norm_sum_m": 0.0,
            "target_norm_count": 0,
        })

    with torch.inference_mode():
        for batch in tqdm(loader, desc=description, dynamic_ncols=True):
            batch = base.dict_apply(batch, lambda x: x.to(args.device, non_blocking=True))
            gt_action = batch["action"].float().reshape(batch["action"].shape[0], batch["action"].shape[1], -1, 10)
            with base.autocast_context(args):
                pred_action = policy.predict_action(batch["obs"], None)["action_pred"]
            pred_action = pred_action.float().reshape(pred_action.shape[0], pred_action.shape[1], -1, 10)
            pos_error = pred_action[..., :3] - gt_action[..., :3]
            gt_pos = gt_action[..., :3]

            for step in range(horizon):
                item = by_step[step]
                this_error = pos_error[:, step]
                this_target = gt_pos[:, step]
                item["error_sq_sum_m2"] += this_error.square().sum(dtype=torch.float64).item()
                item["error_vector_count"] += this_error.shape[0] * this_error.shape[1]
                item["error_abs_sum_m"] += this_error.norm(dim=-1).sum(dtype=torch.float64).item()
                item["target_norm_sum_m"] += this_target.norm(dim=-1).sum(dtype=torch.float64).item()
                item["target_norm_count"] += this_target.shape[0] * this_target.shape[1]

            for name, this_slice in horizon_slices.items():
                this_error = pos_error[:, this_slice]
                this_target = gt_pos[:, this_slice]
                error_norm = this_error.norm(dim=-1)
                target_norm = this_target.norm(dim=-1)
                sums.setdefault(name, {"error_sq_sum_m2": 0.0, "error_abs_sum_m": 0.0, "target_norm_sum_m": 0.0})
                counts[name] = counts.get(name, 0) + this_error.shape[0] * this_error.shape[1] * this_error.shape[2]
                sums[name]["error_sq_sum_m2"] += this_error.square().sum(dtype=torch.float64).item()
                sums[name]["error_abs_sum_m"] += error_norm.sum(dtype=torch.float64).item()
                sums[name]["target_norm_sum_m"] += target_norm.sum(dtype=torch.float64).item()
                # Per action-vector distributions (averaging across robots/horizon within a sample).
                sample_error_norms[name].extend(error_norm.mean(dim=(1, 2)).cpu().tolist())
                sample_target_norms[name].extend(target_norm.mean(dim=(1, 2)).cpu().tolist())

            del batch, gt_action, pred_action, pos_error, gt_pos

    aggregate = {}
    for name in horizon_slices:
        vectors = counts[name]
        error_rmse = (sums[name]["error_sq_sum_m2"] / vectors) ** 0.5
        error_mae_vector = sums[name]["error_abs_sum_m"] / vectors
        target_mean = sums[name]["target_norm_sum_m"] / vectors
        aggregate[name] = {
            "position_axis_rmse_m": (sums[name]["error_sq_sum_m2"] / (vectors * 3)) ** 0.5,
            "position_vector_rmse_m": error_rmse,
            "position_axis_mae_m": None,
            "position_vector_mae_m": error_mae_vector,
            "target_vector_mean_m": target_mean,
            "error_vector_rmse_over_target_mean": error_rmse / target_mean if target_mean else float("nan"),
            "error_vector_mae_over_target_mean": error_mae_vector / target_mean if target_mean else float("nan"),
            "action_vectors": vectors,
            "sample_error_vector_mae_percentiles_m": {
                f"p{p}": float(np.percentile(sample_error_norms[name], p)) for p in (50, 75, 90, 95, 99, 100)
            },
            "sample_target_vector_mean_percentiles_m": {
                f"p{p}": float(np.percentile(sample_target_norms[name], p)) for p in (50, 75, 90, 95, 99, 100)
            },
        }
        # Axis MAE needs a separate sum; derive only from direct per-step data below.

    for item in by_step:
        c = item["error_vector_count"]
        item["position_axis_rmse_m"] = (item["error_sq_sum_m2"] / (c * 3)) ** 0.5
        item["position_vector_rmse_m"] = (item["error_sq_sum_m2"] / c) ** 0.5
        item["position_vector_mae_m"] = item["error_abs_sum_m"] / c
        item["target_vector_mean_m"] = item["target_norm_sum_m"] / item["target_norm_count"]
        item["rmse_over_target_mean"] = item["position_vector_rmse_m"] / item["target_vector_mean_m"] if item["target_vector_mean_m"] else float("nan")
        item["mae_over_target_mean"] = item["position_vector_mae_m"] / item["target_vector_mean_m"] if item["target_vector_mean_m"] else float("nan")
        del item["error_sq_sum_m2"], item["error_vector_count"], item["error_abs_sum_m"], item["target_norm_sum_m"], item["target_norm_count"]

    return {
        "aggregate": aggregate,
        "by_prediction_horizon": by_step,
    }


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=PROJECT_ROOT / "data/outputs/2026.08.26")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/outputs/2026.08.26/cross_eval_mse/absolute_error")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    args = parser.parse_args()
    os.chdir(PROJECT_ROOT)
    args.runs_dir = args.runs_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    checkpoints = [s for s in base.discover_checkpoints(args.runs_dir) if s.domain != "" and s.domain in ("umi", "real")]
    cross = {("real", "umi"), ("umi", "real")}
    checkpoints = [s for s in checkpoints if (s.domain, s.state) in {("real", "no_state"), ("real", "with_state"), ("umi", "no_state"), ("umi", "with_state")}]
    payload = base.load_payload(checkpoints[0].checkpoint)
    base_cfg = copy.deepcopy(payload["cfg"])
    del payload
    datasets = {}
    info = {}
    for domain in ("umi", "real"):
        datasets[domain], info[domain] = base.build_validation_dataset(base_cfg, base.DATASETS[domain], args.val_ratio, args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for checkpoint in checkpoints:
        target_domain = "real" if checkpoint.domain == "umi" else "umi"
        print(f"Loading {checkpoint.domain}/{checkpoint.state} -> {target_domain}", flush=True)
        policy, _ = base.build_policy(checkpoint, "ema_model", device)
        metrics = evaluate(policy, datasets[target_domain], target_domain, args, f"absolute {checkpoint.domain}/{checkpoint.state} -> {target_domain}")
        results.append({
            "model_domain": checkpoint.domain,
            "state": checkpoint.state,
            "data_domain": target_domain,
            "checkpoint": str(checkpoint.checkpoint.relative_to(PROJECT_ROOT)),
            "samples_evaluated": info[target_domain]["validation_samples"],
            **metrics,
        })
        policy.to("cpu")
        del policy
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output = {
        "metadata": {
            "metric": "position absolute/vector errors from EMA action predictions",
            "seed": args.seed,
            "val_ratio": args.val_ratio,
            "batch_size": args.batch_size,
            "precision": args.precision,
            "execution_reference_steps": N_EXECUTED_STEPS,
            "dataset_info": info,
        },
        "results": results,
    }
    (args.output_dir / "absolute_error.json").write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {args.output_dir / 'absolute_error.json'}", flush=True)


if __name__ == "__main__":
    main()
