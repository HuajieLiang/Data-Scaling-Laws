#!/usr/bin/env python3
"""Cross-evaluate UMI and real-robot checkpoints on fixed validation splits.

The metric follows the action sampling evaluation in
``train_diffusion_unet_image_workspace.py``: the EMA policy predicts an
unnormalized 16-step action and MSE is computed in action space.  Unlike the
training logger, this script accumulates squared errors and element counts,
so a short final batch receives the correct weight.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import os
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dill
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from diffusion_policy.common.pytorch_util import dict_apply  # noqa: E402


# Checkpoint configs contain expressions such as the observation latency.
OmegaConf.register_new_resolver("eval", eval, replace=True)


@dataclass(frozen=True)
class DatasetSpec:
    domain: str
    path: Path
    frequency: float


@dataclass(frozen=True)
class CheckpointSpec:
    domain: str
    state: str
    run_dir: Path
    checkpoint: Path


DATASETS = {
    "umi": DatasetSpec(
        domain="umi",
        path=PROJECT_ROOT / "data/dataset_umi_zarr/pick_cube_1cam/dataset.zarr.zip",
        frequency=23.12959389342191,
    ),
    "real": DatasetSpec(
        domain="real",
        path=PROJECT_ROOT / "data/dataset_real_zarr/pick_cube_1cam/dataset.zarr.zip",
        frequency=20.0,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=PROJECT_ROOT / "data/outputs/2026.08.26",
        help="Directory containing the four pick_cube run directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data/outputs/2026.08.26/cross_eval_mse",
    )
    parser.add_argument("--model-domains", nargs="+", choices=("umi", "real"), default=("umi", "real"))
    parser.add_argument("--data-domains", nargs="+", choices=("umi", "real"), default=("umi", "real"))
    parser.add_argument("--states", nargs="+", choices=("no_state", "with_state"), default=("no_state", "with_state"))
    parser.add_argument("--weights", choices=("ema_model", "model"), default="ema_model")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--max-batches", type=int, default=None, help="For smoke tests only.")
    parser.add_argument("--overwrite", action="store_true", help="Discard compatible partial results and rerun cells.")
    return parser.parse_args()


def discover_checkpoints(runs_dir: Path) -> list[CheckpointSpec]:
    found: dict[tuple[str, str], CheckpointSpec] = {}
    for checkpoint in sorted(runs_dir.glob("*/checkpoints/latest.ckpt")):
        run_dir = checkpoint.parent.parent
        name = run_dir.name
        if "pick_cube" not in name or "_1cam_" not in name:
            continue
        if "_umi_" in name:
            domain = "umi"
        elif "_real_" in name:
            domain = "real"
        else:
            continue
        if name.endswith("_no_state"):
            state = "no_state"
        elif name.endswith("_with_state"):
            state = "with_state"
        else:
            continue
        key = (domain, state)
        if key in found:
            raise RuntimeError(f"Multiple checkpoints found for {key}: {found[key].checkpoint} and {checkpoint}")
        found[key] = CheckpointSpec(domain, state, run_dir, checkpoint)
    return [found[key] for key in sorted(found)]


def load_payload(path: Path) -> dict[str, Any]:
    # mmap avoids eagerly materializing both the raw and EMA copies in RAM.
    return torch.load(str(path), map_location="cpu", pickle_module=dill, mmap=True)


def target_dataset_config(base_cfg: DictConfig, spec: DatasetSpec, val_ratio: float, seed: int) -> DictConfig:
    cfg = copy.deepcopy(base_cfg)
    relative_path = os.path.relpath(spec.path, PROJECT_ROOT)
    cfg.task.dataset_path = relative_path
    cfg.task.dataset_frequeny = spec.frequency
    cfg.task.dataset.dataset_path = relative_path
    cfg.task.dataset.val_ratio = val_ratio
    cfg.task.dataset.seed = seed
    cfg.task.dataset.dataset_idx = None
    cfg.task.dataset.use_ratio = 1.0
    return cfg


def build_validation_dataset(base_cfg: DictConfig, spec: DatasetSpec, val_ratio: float, seed: int):
    cfg = target_dataset_config(base_cfg, spec, val_ratio, seed)
    training_dataset = hydra.utils.instantiate(cfg.task.dataset)
    val_episodes = int(training_dataset.val_mask.sum())
    total_episodes = int(training_dataset.val_mask.size)
    val_dataset = training_dataset.get_validation_dataset()
    info = {
        "dataset": str(spec.path.relative_to(PROJECT_ROOT)),
        "frequency_hz": spec.frequency,
        "total_episodes": total_episodes,
        "validation_episodes": val_episodes,
        "validation_samples": len(val_dataset),
    }
    del training_dataset
    return val_dataset, info


def build_policy(spec: CheckpointSpec, weights: str, device: torch.device):
    payload = load_payload(spec.checkpoint)
    cfg = payload["cfg"]
    # All learned encoder parameters are restored below; initializing ImageNet
    # weights here only wastes time and may trigger an unnecessary download.
    cfg.policy.obs_encoder.pretrained = False
    policy = hydra.utils.instantiate(cfg.policy)
    policy.load_state_dict(payload["state_dicts"][weights], strict=True)
    del payload
    gc.collect()
    policy.eval()
    policy.requires_grad_(False)
    return policy.to(device), cfg


def make_loader(dataset, args: argparse.Namespace) -> DataLoader:
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "shuffle": False,
        "pin_memory": args.device.startswith("cuda"),
        "persistent_workers": False,
        "generator": torch.Generator().manual_seed(args.seed),
    }
    if args.num_workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def autocast_context(args: argparse.Namespace):
    if not args.device.startswith("cuda") or args.precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def evaluate(policy, dataset, args: argparse.Namespace, description: str) -> dict[str, Any]:
    seed_everything(args.seed)
    loader = make_loader(dataset, args)
    squared_error = {"overall": 0.0, "position": 0.0, "rotation_6d": 0.0, "gripper_width": 0.0}
    element_count = {key: 0 for key in squared_error}
    samples = 0
    batches = 0
    start_time = time.monotonic()

    with torch.inference_mode():
        progress = tqdm(loader, desc=description, dynamic_ncols=True)
        for batch_index, batch in enumerate(progress):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            batch = dict_apply(batch, lambda x: x.to(args.device, non_blocking=True))
            ground_truth = batch["action"]
            with autocast_context(args):
                prediction = policy.predict_action(batch["obs"], None)["action_pred"]

            prediction = prediction.float().reshape(prediction.shape[0], prediction.shape[1], -1, 10)
            ground_truth = ground_truth.float().reshape(ground_truth.shape[0], ground_truth.shape[1], -1, 10)
            diff = prediction - ground_truth
            slices = {
                "overall": diff,
                "position": diff[..., :3],
                "rotation_6d": diff[..., 3:9],
                "gripper_width": diff[..., 9:10],
            }
            for key, value in slices.items():
                squared_error[key] += value.square().sum(dtype=torch.float64).item()
                element_count[key] += value.numel()
            samples += prediction.shape[0]
            batches += 1
            progress.set_postfix(overall=f"{squared_error['overall'] / element_count['overall']:.6g}")

            del batch, ground_truth, prediction, diff

    elapsed = time.monotonic() - start_time
    return {
        "mse": {key: squared_error[key] / element_count[key] for key in squared_error},
        "squared_error_sum": squared_error,
        "element_count": element_count,
        "samples_evaluated": samples,
        "batches_evaluated": batches,
        "elapsed_seconds": elapsed,
    }


def result_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return row["model_domain"], row["state"], row["data_domain"]


def metadata_for(args: argparse.Namespace, dataset_info: dict[str, Any]) -> dict[str, Any]:
    return {
        "metric": "sample-weighted elementwise MSE on unnormalized 16-step action predictions",
        "weights": args.weights,
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "batch_size": args.batch_size,
        "precision": args.precision,
        "max_batches": args.max_batches,
        "dataset_info": dataset_info,
    }


def compatible_metadata(old: dict[str, Any], new: dict[str, Any]) -> bool:
    keys = ("metric", "weights", "seed", "val_ratio", "batch_size", "precision", "max_batches")
    return all(old.get(key) == new.get(key) for key in keys)


def write_outputs(output_dir: Path, metadata: dict[str, Any], results: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    results = sorted(results, key=result_key)
    json_path = output_dir / "cross_eval_mse.json"
    csv_path = output_dir / "cross_eval_mse.csv"
    markdown_path = output_dir / "cross_eval_mse.md"

    json_path.write_text(
        json.dumps({"metadata": metadata, "results": results}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    fields = [
        "model_domain", "state", "data_domain", "mse_overall", "mse_position",
        "mse_rotation_6d", "mse_gripper_width", "samples_evaluated", "batches_evaluated",
        "checkpoint", "dataset", "elapsed_seconds",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow({
                "model_domain": row["model_domain"],
                "state": row["state"],
                "data_domain": row["data_domain"],
                "mse_overall": row["mse"]["overall"],
                "mse_position": row["mse"]["position"],
                "mse_rotation_6d": row["mse"]["rotation_6d"],
                "mse_gripper_width": row["mse"]["gripper_width"],
                "samples_evaluated": row["samples_evaluated"],
                "batches_evaluated": row["batches_evaluated"],
                "checkpoint": row["checkpoint"],
                "dataset": row["dataset"],
                "elapsed_seconds": row["elapsed_seconds"],
            })

    lines = [
        "# UMI / real-robot cross-evaluation MSE",
        "",
        f"EMA/action-space evaluation; seed={metadata['seed']}, val_ratio={metadata['val_ratio']}, "
        f"precision={metadata['precision']}. Values are global elementwise means over the fixed validation split.",
        "",
        "| Model data | State | Eval data | Overall MSE | Position MSE | Rotation-6D MSE | Gripper-width MSE | Samples |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        mse = row["mse"]
        lines.append(
            f"| {row['model_domain']} | {row['state']} | {row['data_domain']} | "
            f"{mse['overall']:.10g} | {mse['position']:.10g} | {mse['rotation_6d']:.10g} | "
            f"{mse['gripper_width']:.10g} | {row['samples_evaluated']} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    os.chdir(PROJECT_ROOT)
    args.runs_dir = args.runs_dir.resolve()
    args.output_dir = args.output_dir.resolve()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable. Run this command in the GPU-enabled environment.")
    if args.precision == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This CUDA device does not support BF16; use --precision fp16 or fp32.")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    checkpoints = discover_checkpoints(args.runs_dir)
    selected = [
        spec for spec in checkpoints
        if spec.domain in args.model_domains and spec.state in args.states
    ]
    expected = {(domain, state) for domain in args.model_domains for state in args.states}
    actual = {(spec.domain, spec.state) for spec in selected}
    if actual != expected:
        raise RuntimeError(f"Missing checkpoint variants: {sorted(expected - actual)}")
    for spec in selected:
        if not spec.checkpoint.is_file():
            raise FileNotFoundError(spec.checkpoint)
    for domain in args.data_domains:
        if not DATASETS[domain].path.is_file():
            raise FileNotFoundError(DATASETS[domain].path)

    # Any run has the same one-camera dataset schema.  Target-domain frequency
    # and path are applied before instantiating each validation dataset.
    config_payload = load_payload(selected[0].checkpoint)
    base_cfg = copy.deepcopy(config_payload["cfg"])
    del config_payload

    datasets: dict[str, Any] = {}
    dataset_info: dict[str, Any] = {}
    for domain in args.data_domains:
        print(f"Building fixed {domain} validation split...", flush=True)
        datasets[domain], dataset_info[domain] = build_validation_dataset(
            base_cfg, DATASETS[domain], args.val_ratio, args.seed
        )
        print(json.dumps({domain: dataset_info[domain]}, ensure_ascii=False), flush=True)

    metadata = metadata_for(args, dataset_info)
    result_path = args.output_dir / "cross_eval_mse.json"
    results: list[dict[str, Any]] = []
    if result_path.is_file() and not args.overwrite:
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if not compatible_metadata(existing.get("metadata", {}), metadata):
            raise RuntimeError(f"Existing results in {result_path} use incompatible settings; pass --overwrite.")
        results = existing.get("results", [])
    completed = {result_key(row) for row in results}

    for checkpoint_spec in selected:
        pending_domains = [
            domain for domain in args.data_domains
            if (checkpoint_spec.domain, checkpoint_spec.state, domain) not in completed
        ]
        if not pending_domains:
            continue
        print(
            f"Loading {args.weights}: model={checkpoint_spec.domain}/{checkpoint_spec.state} "
            f"from {checkpoint_spec.checkpoint}",
            flush=True,
        )
        policy, _ = build_policy(checkpoint_spec, args.weights, device)
        for data_domain in pending_domains:
            description = f"{checkpoint_spec.domain}/{checkpoint_spec.state} -> {data_domain}"
            metrics = evaluate(policy, datasets[data_domain], args, description)
            row = {
                "model_domain": checkpoint_spec.domain,
                "state": checkpoint_spec.state,
                "data_domain": data_domain,
                "checkpoint": str(checkpoint_spec.checkpoint.relative_to(PROJECT_ROOT)),
                "dataset": str(DATASETS[data_domain].path.relative_to(PROJECT_ROOT)),
                **metrics,
            }
            results.append(row)
            completed.add(result_key(row))
            write_outputs(args.output_dir, metadata, results)
            print(json.dumps(row, ensure_ascii=False), flush=True)

        policy.to("cpu")
        del policy
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_outputs(args.output_dir, metadata, results)
    print(f"Results written to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
