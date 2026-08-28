#!/usr/bin/env python3
"""Measure UMI/real separability in the fine-tuned visual activations.

This uses the EMA DINOv2 observation encoders from the two no-state models.
Only the encoder is materialized, avoiding the much larger diffusion decoder.
The held-out domain classifier is split by episode to prevent adjacent frames
from leaking between train and test.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from pathlib import Path
from typing import Any

import dill
import hydra
import numpy as np
import torch
import zarr
from omegaconf import OmegaConf
from scipy.stats import rankdata


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs  # noqa: E402


register_codecs()
OmegaConf.register_new_resolver("eval", eval, replace=True)

DATASETS = {
    "umi": PROJECT_ROOT / "data/dataset_umi_zarr/pick_cube_1cam/dataset.zarr.zip",
    "real": PROJECT_ROOT / "data/dataset_real_zarr/pick_cube_1cam/dataset.zarr.zip",
}
CHECKPOINTS = {
    "umi": PROJECT_ROOT / "data/outputs/2026.08.26/26-16.56.40_pick_cube_umi_1cam_no_state/checkpoints/latest.ckpt",
    "real": PROJECT_ROOT / "data/outputs/2026.08.26/26-16.59.45_pick_cube_real_1cam_no_state/checkpoints/latest.ckpt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data/outputs/2026.08.26/cross_eval_mse/domain_gap_analysis",
    )
    parser.add_argument("--samples-per-domain", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-threads", type=int, default=16)
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


def select_sequences(path: Path, count: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    with zarr.ZipStore(str(path), mode="r") as store:
        root = zarr.open_group(store, mode="r")
        ends = root["meta/episode_ends"][:]
        starts = np.r_[0, ends[:-1]]
        per_episode = max(1, int(np.ceil(count / len(ends))))
        current_ids = []
        episode_ids = []
        for episode_id, (start, end) in enumerate(zip(starts, ends)):
            candidates = np.arange(start, end)
            selected = rng.choice(candidates, size=min(per_episode, len(candidates)), replace=False)
            current_ids.extend(int(x) for x in selected)
            episode_ids.extend([episode_id] * len(selected))
        if len(current_ids) > count:
            keep = rng.choice(len(current_ids), size=count, replace=False)
            current_ids = [current_ids[i] for i in keep]
            episode_ids = [episode_ids[i] for i in keep]

        sequences = []
        images = root["data/camera0_rgb"]
        for current, episode_id in zip(current_ids, episode_ids):
            start = int(starts[episode_id])
            previous = max(start, current - 3)
            pair = np.stack([images[previous], images[current]], axis=0)
            sequences.append(np.moveaxis(pair, -1, 1).astype(np.float32) / 255.0)
    return np.stack(sequences), np.asarray(episode_ids), np.asarray(current_ids)


def load_encoder(checkpoint: Path):
    payload = torch.load(str(checkpoint), map_location="cpu", pickle_module=dill, mmap=True)
    cfg = payload["cfg"]
    cfg.policy.obs_encoder.pretrained = False
    encoder = hydra.utils.instantiate(cfg.policy.obs_encoder)
    prefix = "obs_encoder."
    encoder_state = {
        key[len(prefix):]: value
        for key, value in payload["state_dicts"]["ema_model"].items()
        if key.startswith(prefix)
    }
    missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Encoder state mismatch: missing={missing}, unexpected={unexpected}")
    del payload
    gc.collect()
    encoder.eval().requires_grad_(False)
    return encoder


def extract(encoder, sequences: np.ndarray, batch_size: int) -> np.ndarray:
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(sequences), batch_size):
            batch = torch.from_numpy(sequences[start:start + batch_size])
            feature = encoder({"camera0_rgb": batch})
            outputs.append(feature.float().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def episode_split(episode_ids: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    unique = rng.permutation(np.unique(episode_ids))
    n_train = min(max(1, round(len(unique) * 0.8)), len(unique) - 1)
    return np.isin(episode_ids, unique[:n_train])


def auc_score(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(bool)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    ranks = rankdata(scores)
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def pca_lda(features: np.ndarray, labels: np.ndarray, train_mask: np.ndarray) -> dict[str, float]:
    train = features[train_mask].astype(np.float64)
    test = features[~train_mask].astype(np.float64)
    y_train = labels[train_mask]
    y_test = labels[~train_mask]
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std < 1e-7] = 1.0
    train = (train - mean) / std
    test = (test - mean) / std
    _, _, vt = np.linalg.svd(train - train.mean(axis=0), full_matrices=False)
    components = min(24, len(vt), vt.shape[1])
    projection = vt[:components].T
    train = train @ projection
    test = test @ projection
    mu0 = train[y_train == 0].mean(axis=0)
    mu1 = train[y_train == 1].mean(axis=0)
    centered = np.concatenate([train[y_train == 0] - mu0, train[y_train == 1] - mu1])
    covariance = centered.T @ centered / max(1, len(centered) - 2)
    regularizer = max(float(np.trace(covariance) / len(covariance)) * 0.1, 1e-5)
    covariance += np.eye(len(covariance)) * regularizer
    direction = np.linalg.solve(covariance, mu1 - mu0)
    threshold = 0.5 * float((mu0 + mu1) @ direction)
    scores = test @ direction - threshold
    prediction = scores >= 0
    acc0 = float(np.mean(prediction[y_test == 0] == 0))
    acc1 = float(np.mean(prediction[y_test == 1] == 1))
    auc = auc_score(y_test, scores)
    if auc < 0.5:
        auc = 1.0 - auc
    return {
        "balanced_accuracy": (acc0 + acc1) / 2,
        "auc": auc,
        "test_samples": int(len(test)),
        "pca_components": components,
    }


def geometry(features: np.ndarray, labels: np.ndarray, source_label: int) -> dict[str, float]:
    x0 = features[labels == 0].astype(np.float64)
    x1 = features[labels == 1].astype(np.float64)
    mu0 = x0.mean(axis=0)
    mu1 = x1.mean(axis=0)
    rms0 = float(np.sqrt(np.mean(np.sum(np.square(x0 - mu0), axis=-1))))
    rms1 = float(np.sqrt(np.mean(np.sum(np.square(x1 - mu1), axis=-1))))
    centroid_distance = float(np.linalg.norm(mu1 - mu0))
    source = x0 if source_label == 0 else x1
    target = x1 if source_label == 0 else x0
    source_centroid = source.mean(axis=0)
    source_distance = np.linalg.norm(source - source_centroid, axis=-1)
    target_distance = np.linalg.norm(target - source_centroid, axis=-1)
    cosine = float(mu0 @ mu1 / max(np.linalg.norm(mu0) * np.linalg.norm(mu1), 1e-12))
    return {
        "centroid_distance": centroid_distance,
        "within_domain_rms_umi": rms0,
        "within_domain_rms_real": rms1,
        "centroid_distance_over_mean_within_rms": centroid_distance / max((rms0 + rms1) / 2, 1e-12),
        "target_to_source_centroid_mean_over_source": float(target_distance.mean() / source_distance.mean()),
        "domain_centroid_cosine": cosine,
    }


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.num_threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    sequences = {}
    episode_ids = {}
    frame_ids = {}
    for index, domain in enumerate(("umi", "real")):
        sequences[domain], episode_ids[domain], frame_ids[domain] = select_sequences(
            DATASETS[domain], args.samples_per_domain, args.seed + index
        )

    labels = np.concatenate([
        np.zeros(len(sequences["umi"]), dtype=np.int64),
        np.ones(len(sequences["real"]), dtype=np.int64),
    ])
    train_mask = np.concatenate([
        episode_split(episode_ids["umi"], args.seed + 10),
        episode_split(episode_ids["real"], args.seed + 11),
    ])

    results = []
    for source_label, model_domain in enumerate(("umi", "real")):
        print(f"Loading {model_domain} no-state encoder", flush=True)
        encoder = load_encoder(CHECKPOINTS[model_domain])
        domain_features = []
        for data_domain in ("umi", "real"):
            print(f"Extracting {model_domain} encoder on {data_domain} images", flush=True)
            domain_features.append(extract(encoder, sequences[data_domain], args.batch_size))
        features = np.concatenate(domain_features, axis=0)
        result = {
            "model_domain": model_domain,
            "checkpoint": str(CHECKPOINTS[model_domain].relative_to(PROJECT_ROOT)),
            "activation_shape": list(features.shape),
            "domain_classifier": pca_lda(features, labels, train_mask),
            "activation_geometry": geometry(features, labels, source_label),
        }
        results.append(result)
        del encoder, features, domain_features
        gc.collect()

    report = {
        "metadata": {
            "weights": "ema_model",
            "models": "no_state",
            "samples_per_domain": args.samples_per_domain,
            "observation_frames": 2,
            "observation_stride_frames": 3,
            "split": "held-out episodes",
            "device": "cpu",
        },
        "results": results,
    }
    output = args.output_dir / "activation_domain_gap.json"
    output.write_text(json.dumps(native(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(native(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
