#!/usr/bin/env python3
"""Verify that cached WiLoR MANO parameters reproduce cached hand vertices."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wilor.models import load_wilor  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-npz", required=True)
    parser.add_argument(
        "--checkpoint", default="pretrained_models/wilor_final.ckpt"
    )
    parser.add_argument(
        "--config", default="pretrained_models/model_config.yaml"
    )
    parser.add_argument("--mano-data-dir", default=None)
    parser.add_argument("--out-json")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-rmse-mm", type=float, default=0.1)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def distribution(values: np.ndarray) -> dict[str, float | int | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {"count": 0, "median": None, "p90": None, "max": None}
    return {
        "count": int(len(finite)),
        "median": float(np.median(finite)),
        "p90": float(np.percentile(finite, 90)),
        "max": float(np.max(finite)),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    query_path = resolve_path(args.query_npz)
    query = load_npz(query_path)
    required = {
        "model_valid",
        "vertices_3d_root_relative_canonical_right",
        "vertices_3d_root_relative_original",
        "mano_global_orient_canonical_right",
        "mano_hand_pose_canonical_right",
        "mano_betas",
        "hand_side",
    }
    missing = sorted(required - set(query))
    if missing:
        raise KeyError(f"Query cache lacks {missing}; re-export with cache v2")

    valid = np.asarray(query["model_valid"]).astype(bool)
    parameter_finite = (
        np.isfinite(query["mano_global_orient_canonical_right"])
        .reshape(len(valid), -1).all(axis=1)
        & np.isfinite(query["mano_hand_pose_canonical_right"])
        .reshape(len(valid), -1).all(axis=1)
        & np.isfinite(query["mano_betas"]).reshape(len(valid), -1).all(axis=1)
    )
    valid &= parameter_finite
    indices = np.flatnonzero(valid)
    if not len(indices):
        raise RuntimeError("No valid frames with finite MANO parameters")

    model, _ = load_wilor(
        checkpoint_path=str(resolve_path(args.checkpoint)),
        cfg_path=str(resolve_path(args.config)),
        init_renderer=False,
        mano_data_dir=(
            str(resolve_path(args.mano_data_dir))
            if args.mano_data_dir else None
        ),
    )
    device = torch.device(args.device)
    mano = model.mano.to(device).eval()
    del model

    reconstructed = np.full_like(
        query["vertices_3d_root_relative_canonical_right"], np.nan,
        dtype=np.float32,
    )
    with torch.inference_mode():
        for start in range(0, len(indices), args.batch_size):
            batch_indices = indices[start:start + args.batch_size]
            global_orient = torch.from_numpy(np.asarray(
                query["mano_global_orient_canonical_right"][batch_indices],
                dtype=np.float32,
            )).to(device)
            hand_pose = torch.from_numpy(np.asarray(
                query["mano_hand_pose_canonical_right"][batch_indices],
                dtype=np.float32,
            )).to(device)
            betas = torch.from_numpy(np.asarray(
                query["mano_betas"][batch_indices], dtype=np.float32,
            )).to(device)
            output = mano(
                global_orient=global_orient,
                hand_pose=hand_pose,
                betas=betas,
                pose2rot=False,
            )
            local = output.vertices - output.joints[:, :1]
            reconstructed[batch_indices] = local.float().cpu().numpy()

    canonical_target = np.asarray(
        query["vertices_3d_root_relative_canonical_right"], dtype=np.float32
    )
    original_target = np.asarray(
        query["vertices_3d_root_relative_original"], dtype=np.float32
    )
    reconstructed_original = reconstructed.copy()
    if str(query["hand_side"].item()).lower() == "left":
        reconstructed_original[..., 0] *= -1.0

    canonical_error = np.linalg.norm(
        reconstructed[indices] - canonical_target[indices], axis=-1
    ) * 1000.0
    original_error = np.linalg.norm(
        reconstructed_original[indices] - original_target[indices], axis=-1
    ) * 1000.0
    canonical_rmse = np.sqrt(np.mean(np.square(canonical_error), axis=-1))
    original_rmse = np.sqrt(np.mean(np.square(original_error), axis=-1))
    rotation_orthogonality = []
    rotation_determinant = []
    for key in (
        "mano_global_orient_canonical_right",
        "mano_hand_pose_canonical_right",
    ):
        rotations = np.asarray(query[key][indices], dtype=np.float64)
        identity_error = rotations @ np.swapaxes(rotations, -1, -2) - np.eye(3)
        rotation_orthogonality.append(
            np.linalg.norm(identity_error, axis=(-2, -1)).reshape(-1)
        )
        rotation_determinant.append(np.linalg.det(rotations).reshape(-1))

    summary = {
        "query_npz": str(query_path),
        "cache_version": (
            str(query["cache_version"].item())
            if "cache_version" in query else "unknown"
        ),
        "hand_side": str(query["hand_side"].item()),
        "frames": int(len(valid)),
        "valid_frames": int(len(indices)),
        "canonical_vertex_error_mm": distribution(canonical_error),
        "canonical_frame_rmse_mm": distribution(canonical_rmse),
        "original_vertex_error_mm": distribution(original_error),
        "original_frame_rmse_mm": distribution(original_rmse),
        "rotation_orthogonality_error": distribution(
            np.concatenate(rotation_orthogonality)
        ),
        "rotation_determinant": distribution(
            np.concatenate(rotation_determinant)
        ),
        "threshold_rmse_mm": args.max_rmse_mm,
        "passed": bool(np.max(original_rmse) <= args.max_rmse_mm),
    }
    if args.out_json:
        write_json(resolve_path(args.out_json), summary)
    print(json.dumps(summary, indent=2))
    if not summary["passed"]:
        raise RuntimeError(
            "MANO roundtrip exceeded threshold: "
            f"{np.max(original_rmse):.6f} > {args.max_rmse_mm:.6f} mm"
        )


if __name__ == "__main__":
    main()
