#!/usr/bin/env python3
"""Create deterministic mixed pick_cube 1-camera UMI Zarr datasets.

The source files are Zarr v2 ZIP stores using the ``imagecodecs_jpegxl``
codec.  Each output is allocated at its final size
before any data are copied; this avoids ZipStore resize/rename limitations and
keeps each image chunk written exactly once.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import zarr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UMI = PROJECT_ROOT / "data/dataset_umi_zarr/pick_cube_1cam/dataset.zarr.zip"
DEFAULT_REAL = PROJECT_ROOT / "data/dataset_real_zarr/pick_cube_1cam/dataset.zarr.zip"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data/dataset_mix_zarr"


def open_source(path: Path):
    """Open a source store after registering its JPEGXL codec."""
    import imagecodecs.numcodecs

    imagecodecs.numcodecs.register_codecs(verbose=False)
    store = zarr.ZipStore(str(path), mode="r")
    return store, zarr.open_group(store=store, mode="r")


def episode_ranges(episode_ends: np.ndarray, limit: int | None = None):
    selected = len(episode_ends) if limit is None else min(limit, len(episode_ends))
    previous = 0
    for index, end in enumerate(episode_ends[:selected]):
        end = int(end)
        yield index, previous, end
        previous = end


def source_info(path: Path, root: zarr.Group, selected_episodes: int | None) -> dict:
    ends = np.asarray(root["meta/episode_ends"][:], dtype=np.int64)
    selected = list(episode_ranges(ends, selected_episodes))
    frames = int(selected[-1][2]) if selected else 0
    return {
        "path": str(path.resolve()),
        "full_episodes": int(len(ends)),
        "full_frames": int(ends[-1]) if len(ends) else 0,
        "selected_episodes": len(selected),
        "selected_frames": frames,
        "selected_episode_indices": [item[0] for item in selected],
    }


def validate_compatible(umi: zarr.Group, real: zarr.Group) -> None:
    umi_keys = sorted(umi["data"].array_keys())
    real_keys = sorted(real["data"].array_keys())
    if umi_keys != real_keys:
        raise ValueError(f"data keys differ: UMI={umi_keys}, real={real_keys}")
    for key in umi_keys:
        left, right = umi["data"][key], real["data"][key]
        if left.shape[1:] != right.shape[1:]:
            raise ValueError(f"{key}: trailing shapes differ: {left.shape} vs {right.shape}")
        if left.dtype != right.dtype:
            raise ValueError(f"{key}: dtypes differ: {left.dtype} vs {right.dtype}")
        if left.chunks != right.chunks:
            raise ValueError(f"{key}: chunks differ: {left.chunks} vs {right.chunks}")
        left_codec = None if left.compressor is None else left.compressor.get_config()
        right_codec = None if right.compressor is None else right.compressor.get_config()
        if left_codec != right_codec:
            raise ValueError(f"{key}: codec configurations differ")


def copy_source(
    output_arrays: dict[str, zarr.Array],
    source: zarr.Group,
    source_start: int,
    source_end: int,
    output_start: int,
    batch_size: int,
) -> None:
    """Remap already-compressed image chunks without lossy re-encoding."""
    length = source_end - source_start
    for key, output_array in output_arrays.items():
        source_array = source["data"][key]
        if source_array.chunks[0] != 1 or output_array.chunks[0] != 1:
            raise ValueError(f"{key}: raw chunk remapping requires time chunk size 1")
        for offset in range(0, length, batch_size):
            count = min(batch_size, length - offset)
            for inner in range(count):
                src_index = source_start + offset + inner
                dst_index = output_start + offset + inner
                src_key = source_array._chunk_key((src_index, 0, 0, 0))
                dst_key = output_array._chunk_key((dst_index, 0, 0, 0))
                output_array.chunk_store[dst_key] = source_array.chunk_store[src_key]


def build_mix(
    *,
    name: str,
    output_root: Path,
    umi_path: Path,
    real_path: Path,
    parts: Iterable[tuple[str, int | None]],
    batch_size: int,
    overwrite: bool,
) -> dict:
    source_stores = {}
    try:
        source_stores["umi"] = open_source(umi_path)
        source_stores["real"] = open_source(real_path)
        umi = source_stores["umi"][1]
        real = source_stores["real"][1]
        validate_compatible(umi, real)

        plans = []
        for source_name, episode_limit in parts:
            source_path = umi_path if source_name == "umi" else real_path
            source_root = umi if source_name == "umi" else real
            info = source_info(source_path, source_root, episode_limit)
            selected_ranges = list(
                episode_ranges(
                    np.asarray(source_root["meta/episode_ends"][:], dtype=np.int64),
                    episode_limit,
                )
            )
            plans.append({"source": source_name, "info": info, "ranges": selected_ranges})

        total_episodes = sum(len(plan["ranges"]) for plan in plans)
        total_frames = sum(
            int(plan["ranges"][-1][2]) if plan["ranges"] else 0 for plan in plans
        )
        output_dir = output_root / name
        output_path = output_dir / "dataset.zarr.zip"
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"output exists (use --overwrite): {output_path}")
        output_dir.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            output_path.unlink()

        # Both source stores carry the same canonical field contract.  Keep the
        # UMI attrs as the base, then explicitly identify this as a mixed source.
        attrs = dict(umi.attrs.asdict())
        convention = dict(attrs.get("coordinate_convention", {}))
        convention["source"] = "mixed UMI and real-robot datasets"
        convention["input_pose_frame"] = "tcp (already converted in each source Zarr)"
        convention["tracked_to_tcp_applied"] = False
        convention["tracked_to_tcp_matrix"] = np.eye(4).tolist()
        convention["composition"] = (
            "source Zarr already stores T_D_TCP(t); mixing only concatenates episodes"
        )
        convention["tcp_geometry_assumption"] = (
            "real TCP geometry is identical to the canonical UMI TCP"
        )
        attrs["coordinate_convention"] = convention
        attrs["dataset_mix"] = {
            "strategy": name,
            "selection_rule": "episode indices are deterministic prefixes (0..N-1) in each source Zarr",
            "order": [source_name for source_name, _ in parts],
            "sources": [plan["info"] for plan in plans],
        }

        with zarr.ZipStore(str(output_path), mode="w") as output_store:
            output = zarr.group(store=output_store)
            output.attrs.update(attrs)
            meta = output.require_group("meta")
            data = output.require_group("data")

            output_ends = []
            cumulative = 0
            for plan in plans:
                for _, start, end in plan["ranges"]:
                    cumulative += end - start
                    output_ends.append(cumulative)
            if cumulative != total_frames or len(output_ends) != total_episodes:
                raise AssertionError("internal episode plan total mismatch")

            episode_array = meta.array(
                "episode_ends",
                data=np.asarray(output_ends, dtype=np.int64),
                chunks=(max(1, total_episodes),),
                compressor=None,
            )
            del episode_array

            first_source = umi["data"]
            output_arrays = {}
            for key in sorted(first_source.array_keys()):
                source_array = first_source[key]
                if key == "camera0_rgb":
                    output_arrays[key] = data.zeros(
                        name=key,
                        shape=(total_frames,) + source_array.shape[1:],
                        chunks=source_array.chunks,
                        dtype=source_array.dtype,
                        compressor=source_array.compressor,
                    )
                    continue

                # Low-dimensional chunks span episode boundaries.  Assemble
                # them in memory and write once so ZipStore never receives a
                # duplicate chunk member.
                pieces = []
                for plan in plans:
                    source_group = umi if plan["source"] == "umi" else real
                    if plan["ranges"]:
                        source_end = plan["ranges"][-1][2]
                        pieces.append(source_group["data"][key][:source_end])
                combined = np.concatenate(pieces, axis=0)
                if combined.shape[0] != total_frames:
                    raise AssertionError(f"{key}: low-dimensional total mismatch")
                data.array(
                    name=key,
                    data=combined,
                    chunks=source_array.chunks,
                    compressor=source_array.compressor,
                )

            destination = 0
            episode_counter = 0
            for plan in plans:
                source_group = umi if plan["source"] == "umi" else real
                for source_episode, start, end in plan["ranges"]:
                    copy_source(
                        output_arrays,
                        source_group,
                        start,
                        end,
                        destination,
                        batch_size,
                    )
                    destination += end - start
                    episode_counter += 1
                    if episode_counter == 1 or episode_counter % 10 == 0 or episode_counter == total_episodes:
                        print(
                            f"[{name}] copied episode {episode_counter}/{total_episodes} "
                            f"({destination}/{total_frames} frames)",
                            flush=True,
                        )
            if destination != total_frames:
                raise AssertionError("internal frame total mismatch")

        conversion = {
            "conversion_complete": True,
            "output_format": "Data-Scaling-Laws mixed replay-buffer Zarr v2 ZIP",
            "strategy": name,
            "selection_rule": "first N episodes (indices 0..N-1) from each source, in source Zarr order",
            "order": [source_name for source_name, _ in parts],
            "episodes": total_episodes,
            "frames": total_frames,
            "episode_ends": output_ends,
            "camera": "camera0_rgb (1 camera, 224x224x3 RGB)",
            "sources": [plan["info"] for plan in plans],
            "coordinate_convention": convention,
        }
        (output_dir / "conversion.json").write_text(
            json.dumps(conversion, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (output_dir / "count.txt").write_text(
            "".join(f"{len(plan['ranges'])}\n" for plan in plans),
            encoding="utf-8",
        )
        return conversion
    finally:
        for store, _ in source_stores.values():
            store.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--umi", type=Path, default=DEFAULT_UMI)
    parser.add_argument("--real", type=Path, default=DEFAULT_REAL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    for path in (args.umi, args.real):
        if not path.is_file():
            raise FileNotFoundError(path)

    strategies = [
        ("pick_cube_1cam_real20_umi_all", (("real", 20), ("umi", None))),
        ("pick_cube_1cam_umi20_real_all", (("umi", 20), ("real", None))),
        ("pick_cube_1cam_real_all_umi_all", (("real", None), ("umi", None))),
    ]
    for name, parts in strategies:
        build_mix(
            name=name,
            output_root=args.output_root,
            umi_path=args.umi,
            real_path=args.real,
            parts=parts,
            batch_size=args.batch_size,
            overwrite=args.overwrite,
        )
        print(f"[{name}] complete", flush=True)


if __name__ == "__main__":
    main()
