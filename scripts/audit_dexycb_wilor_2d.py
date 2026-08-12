#!/usr/bin/env python3
"""Audit WiLoR's projected 2D joints on one raw DexYCB sequence."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wilor.datasets.vitdet_dataset import ViTDetDataset  # noqa: E402
from wilor.models import load_wilor  # noqa: E402
from wilor.utils import (  # noqa: E402
    crop_normalized_to_full_image,
    full_image_to_crop_normalized,
    recursive_to,
)


JOINT_NAMES = (
    "wrist",
    "thumb_mcp",
    "thumb_pip",
    "thumb_dip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
)
SKELETON_EDGES = tuple(
    (0 if offset == 0 else start + offset - 1, start + offset)
    for start in (1, 5, 9, 13, 17)
    for offset in range(4)
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare WiLoR MANO-projected joints with raw full-image DexYCB "
            "2D annotations. GT-derived boxes isolate keypoint projection from "
            "detector errors."
        )
    )
    parser.add_argument("--sequence-dir", required=True)
    parser.add_argument(
        "--checkpoint", default="pretrained_models/wilor_final.ckpt"
    )
    parser.add_argument(
        "--config", default="pretrained_models/model_config.yaml"
    )
    parser.add_argument(
        "--mano-data-dir",
        default=None,
        help="Directory containing MANO_RIGHT.pkl; defaults to repo mano_data.",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--hand-side",
        choices=("auto", "left", "right"),
        default="auto",
    )
    parser.add_argument("--rescale-factor", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--overlay-count", type=int, default=12)
    return parser.parse_args()


def resolve_path(path_string: str) -> Path:
    path = Path(path_string).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def frame_id(path: Path) -> str:
    return path.stem.rsplit("_", 1)[-1].zfill(6)


def read_hand_side(sequence_dir: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    candidates = [sequence_dir.parent / "meta.yml"]
    candidates.extend(parent / "meta.yml" for parent in sequence_dir.parents[:4])
    for path in candidates:
        if not path.is_file():
            continue
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        sides = payload.get("mano_sides") or payload.get("hand_sides") or []
        if sides:
            side = str(sides[0]).lower()
            if side in {"left", "right"}:
                return side
    raise RuntimeError(
        "Could not infer hand side from meta.yml; pass --hand-side left/right"
    )


def load_gt(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        joints = np.asarray(archive.get("joint_2d", []), dtype=np.float32)
    joints = joints.reshape(-1, 2)
    if joints.shape != (21, 2):
        raise ValueError(f"Unexpected joint_2d shape {joints.shape} in {path}")
    valid = (
        np.isfinite(joints).all(axis=-1)
        & ~np.all(np.isclose(joints, -1.0), axis=-1)
    )
    return joints, valid


def tight_box(joints: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, float]:
    if int(valid.sum()) < 3:
        raise ValueError("Fewer than three valid joints")
    selected = joints[valid]
    lower = selected.min(axis=0)
    upper = selected.max(axis=0)
    center = 0.5 * (lower + upper)
    extent = np.maximum(upper - lower, 20.0)
    box = np.asarray(
        [
            center[0] - extent[0] / 2.0,
            center[1] - extent[1] / 2.0,
            center[0] + extent[0] / 2.0,
            center[1] + extent[1] / 2.0,
        ],
        dtype=np.float32,
    )
    return box, float(extent.max())


def build_records(
    sequence_dir: Path,
    max_frames: int,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    images = sorted(sequence_dir.glob("color_*.jpg"))
    if not images:
        images = sorted(sequence_dir.glob("color_*.png"))
    if not images:
        raise FileNotFoundError(f"No color_*.jpg/png frames in {sequence_dir}")

    records: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for image_path in images:
        identifier = frame_id(image_path)
        label_path = sequence_dir / f"labels_{identifier}.npz"
        try:
            if not label_path.is_file():
                raise FileNotFoundError(label_path)
            joints, valid = load_gt(label_path)
            box, box_size = tight_box(joints, valid)
            records.append(
                {
                    "frame_id": identifier,
                    "image_path": image_path,
                    "gt": joints,
                    "valid": valid,
                    "box": box,
                    "tight_box_size": box_size,
                }
            )
        except Exception as error:
            skipped.append({"frame_id": identifier, "reason": repr(error)})
        if max_frames > 0 and len(records) >= max_frames:
            break
    if not records:
        raise RuntimeError("No frames have usable DexYCB 2D annotations")
    return records, skipped


class DexYCBWiLoRDataset(Dataset):
    def __init__(
        self,
        cfg: Any,
        records: list[dict[str, Any]],
        hand_side: str,
        rescale_factor: float,
    ) -> None:
        self.cfg = cfg
        self.records = records
        self.right = np.float32(hand_side == "right")
        self.rescale_factor = float(rescale_factor)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image = cv2.imread(str(record["image_path"]), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot read {record['image_path']}")
        crop = ViTDetDataset(
            self.cfg,
            image,
            record["box"][None],
            np.asarray([self.right], dtype=np.float32),
            rescale_factor=self.rescale_factor,
        )[0]
        crop.update(
            {
                "frame_id": record["frame_id"],
                "image_path": str(record["image_path"]),
                "gt_keypoints_2d_full": record["gt"],
                "gt_joint_valid": record["valid"],
                "tight_box": record["box"],
                "tight_box_size": np.float32(record["tight_box_size"]),
            }
        )
        return crop


def distribution(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0, "mean": None, "median": None, "p90": None, "max": None}
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "max": float(values.max()),
    }


def summarize(
    predicted: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    box_size: np.ndarray,
) -> dict[str, Any]:
    error = np.linalg.norm(predicted - target, axis=-1)
    normalized = error / np.maximum(box_size[:, None], 1e-6)
    valid_error = error[valid]
    valid_normalized = normalized[valid]
    wrist_valid = valid[:, 0]
    wrist_error = error[wrist_valid, 0]
    wrist_normalized_error = normalized[wrist_valid, 0]
    wrist_aligned_valid = valid & wrist_valid[:, None]
    wrist_aligned_prediction = (
        predicted - predicted[:, :1] + target[:, :1]
    )
    wrist_aligned_error = np.linalg.norm(
        wrist_aligned_prediction - target, axis=-1
    )
    valid_wrist_aligned_error = wrist_aligned_error[wrist_aligned_valid]
    valid_wrist_aligned_normalized = (
        wrist_aligned_error / np.maximum(box_size[:, None], 1e-6)
    )[wrist_aligned_valid]
    similarity_aligned_error = np.full_like(error, np.nan, dtype=np.float64)
    for index in range(len(error)):
        frame_valid = valid[index]
        if int(frame_valid.sum()) < 3:
            continue
        source = predicted[index, frame_valid].astype(np.float64)
        destination = target[index, frame_valid].astype(np.float64)
        source_centered = source - source.mean(axis=0, keepdims=True)
        destination_centered = (
            destination - destination.mean(axis=0, keepdims=True)
        )
        covariance = source_centered.T @ destination_centered
        left, singular, right = np.linalg.svd(covariance)
        correction = np.eye(2, dtype=np.float64)
        correction[-1, -1] = np.sign(
            np.linalg.det(left @ right)
        )
        rotation = left @ correction @ right
        denominator = float(np.sum(np.square(source_centered)))
        scale = float(np.sum(singular * np.diag(correction))) / max(
            denominator, 1e-12
        )
        aligned = (
            scale * source_centered @ rotation
            + destination.mean(axis=0, keepdims=True)
        )
        similarity_aligned_error[index, frame_valid] = np.linalg.norm(
            aligned - destination, axis=-1
        )
    valid_similarity_error = similarity_aligned_error[valid]
    valid_similarity_normalized = (
        similarity_aligned_error / np.maximum(box_size[:, None], 1e-6)
    )[valid]
    frame_error = np.full(len(error), np.nan, dtype=np.float64)
    for index in range(len(error)):
        if valid[index].any():
            frame_error[index] = np.median(error[index, valid[index]])

    per_joint = {}
    for index, name in enumerate(JOINT_NAMES):
        per_joint[name] = distribution(error[valid[:, index], index])
    return {
        "pixel_error": distribution(valid_error),
        "bbox_normalized_error": distribution(valid_normalized),
        "wrist_pixel_error": distribution(wrist_error),
        "wrist_bbox_normalized_error": distribution(wrist_normalized_error),
        "wrist_aligned_pixel_error": distribution(valid_wrist_aligned_error),
        "wrist_aligned_bbox_normalized_error": distribution(
            valid_wrist_aligned_normalized
        ),
        "similarity_aligned_pixel_error": distribution(
            valid_similarity_error
        ),
        "similarity_aligned_bbox_normalized_error": distribution(
            valid_similarity_normalized
        ),
        "pck": {
            "5px": float(np.mean(valid_error <= 5.0)),
            "10px": float(np.mean(valid_error <= 10.0)),
            "20px": float(np.mean(valid_error <= 20.0)),
            "bbox_0.025": float(np.mean(valid_normalized <= 0.025)),
            "bbox_0.05": float(np.mean(valid_normalized <= 0.05)),
            "bbox_0.10": float(np.mean(valid_normalized <= 0.10)),
        },
        "frame_median_pixel_error": distribution(frame_error),
        "per_joint_pixel_error": per_joint,
        "error": error,
        "frame_error": frame_error,
    }


def draw_skeleton(
    image: np.ndarray,
    joints: np.ndarray,
    valid: np.ndarray,
    color: tuple[int, int, int],
) -> None:
    for first, second in SKELETON_EDGES:
        if valid[first] and valid[second]:
            cv2.line(
                image,
                tuple(np.rint(joints[first]).astype(int)),
                tuple(np.rint(joints[second]).astype(int)),
                color,
                2,
                cv2.LINE_AA,
            )
    for point, is_valid in zip(joints, valid):
        if is_valid:
            cv2.circle(
                image,
                tuple(np.rint(point).astype(int)),
                3,
                color,
                -1,
                cv2.LINE_AA,
            )


def save_overlays(
    out_dir: Path,
    frame_ids: list[str],
    image_paths: list[str],
    target: np.ndarray,
    predicted: np.ndarray,
    valid: np.ndarray,
    boxes: np.ndarray,
    frame_error: np.ndarray,
    count: int,
) -> list[str]:
    overlay_dir = out_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    finite = np.flatnonzero(np.isfinite(frame_error))
    order = finite[np.argsort(frame_error[finite])[::-1]][: max(0, count)]
    outputs = []
    for index in order:
        image = cv2.imread(image_paths[index], cv2.IMREAD_COLOR)
        if image is None:
            continue
        draw_skeleton(image, target[index], valid[index], (0, 200, 0))
        draw_skeleton(image, predicted[index], valid[index], (0, 0, 255))
        x1, y1, x2, y2 = np.rint(boxes[index]).astype(int)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 180, 0), 1)
        cv2.putText(
            image,
            f"GT green | WiLoR red | median {frame_error[index]:.1f}px",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 20, 20),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            f"GT green | WiLoR red | median {frame_error[index]:.1f}px",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        output = overlay_dir / (
            f"{frame_ids[index]}_median_{frame_error[index]:.1f}px.jpg"
        )
        cv2.imwrite(str(output), image)
        outputs.append(str(output))
    return outputs


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def main() -> None:
    args = parse_args()
    if args.rescale_factor <= 0:
        raise ValueError("--rescale-factor must be positive")
    sequence_dir = resolve_path(args.sequence_dir)
    checkpoint = resolve_path(args.checkpoint)
    config = resolve_path(args.config)
    mano_data_dir = (
        resolve_path(args.mano_data_dir) if args.mano_data_dir else None
    )
    out_dir = resolve_path(args.out_dir)
    if not sequence_dir.is_dir():
        raise FileNotFoundError(sequence_dir)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not config.is_file():
        raise FileNotFoundError(config)
    if mano_data_dir is not None and not (
        mano_data_dir / "MANO_RIGHT.pkl"
    ).is_file():
        raise FileNotFoundError(mano_data_dir / "MANO_RIGHT.pkl")
    out_dir.mkdir(parents=True, exist_ok=True)

    hand_side = read_hand_side(sequence_dir, args.hand_side)
    records, skipped = build_records(sequence_dir, args.max_frames)
    os.chdir(REPO_ROOT)
    model, cfg = load_wilor(
        str(checkpoint),
        str(config),
        init_renderer=False,
        mano_data_dir=mano_data_dir,
    )
    device = torch.device(args.device)
    model = model.to(device).eval()

    dataset = DexYCBWiLoRDataset(
        cfg, records, hand_side, args.rescale_factor
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    frame_ids: list[str] = []
    image_paths: list[str] = []
    predictions: dict[str, list[np.ndarray]] = {"prior": [], "query": []}
    targets: list[np.ndarray] = []
    valid_masks: list[np.ndarray] = []
    boxes: list[np.ndarray] = []
    tight_sizes: list[np.ndarray] = []
    target_crop_chunks: list[np.ndarray] = []
    prediction_crop_chunks: list[np.ndarray] = []
    coordinate_roundtrip_errors: list[np.ndarray] = []

    with torch.inference_mode():
        for batch in loader:
            frame_ids.extend(batch.pop("frame_id"))
            image_paths.extend(batch.pop("image_path"))
            target = batch.pop("gt_keypoints_2d_full")
            valid = batch.pop("gt_joint_valid").bool()
            box = batch.pop("tight_box")
            tight_size = batch.pop("tight_box_size")
            target_crop = full_image_to_crop_normalized(
                target,
                batch["crop_transform"],
                cfg.MODEL.IMAGE_SIZE,
                batch["img_size"][:, 0],
                batch["right"],
            )
            target_roundtrip = crop_normalized_to_full_image(
                target_crop,
                batch["crop_transform"],
                cfg.MODEL.IMAGE_SIZE,
                batch["img_size"][:, 0],
                batch["right"],
            )
            coordinate_roundtrip_errors.append(
                torch.linalg.vector_norm(target_roundtrip - target, dim=-1)
                .cpu()
                .numpy()
            )
            batch = recursive_to(batch, device)
            output = model(batch)
            prediction_crop_chunks.append(
                output["pred_keypoints_2d_prior"][:, :21].cpu().numpy()
            )

            for name, output_key in (
                ("prior", "pred_keypoints_2d_prior"),
                ("query", "pred_keypoints_2d_query"),
            ):
                full = crop_normalized_to_full_image(
                    output[output_key][:, :21],
                    batch["crop_transform"],
                    cfg.MODEL.IMAGE_SIZE,
                    batch["img_size"][:, 0],
                    batch["right"],
                )
                predictions[name].append(full.cpu().numpy())
            targets.append(target.numpy())
            target_crop_chunks.append(target_crop.numpy())
            valid_masks.append(valid.numpy())
            boxes.append(box.numpy())
            tight_sizes.append(tight_size.numpy())

    target = np.concatenate(targets)
    valid = np.concatenate(valid_masks).astype(bool)
    tight_box = np.concatenate(boxes)
    tight_size = np.concatenate(tight_sizes)
    predicted = {
        name: np.concatenate(chunks) for name, chunks in predictions.items()
    }
    target_crop = np.concatenate(target_crop_chunks)
    prediction_crop = np.concatenate(prediction_crop_chunks)
    roundtrip_error = np.concatenate(coordinate_roundtrip_errors)
    roundtrip_max = float(roundtrip_error[valid].max())
    if roundtrip_max > 1e-3:
        raise RuntimeError(
            "Full-image/crop coordinate roundtrip exceeded 1e-3 px: "
            f"{roundtrip_max:.6f} px"
        )
    for name, value in predicted.items():
        if not np.isfinite(value).all():
            raise RuntimeError(f"Non-finite {name} 2D predictions")
    summaries = {
        name: summarize(value, target, valid, tight_size)
        for name, value in predicted.items()
    }

    prior_summary = summaries["prior"]
    overlays = save_overlays(
        out_dir,
        frame_ids,
        image_paths,
        target,
        predicted["prior"],
        valid,
        tight_box,
        prior_summary["frame_error"],
        args.overlay_count,
    )

    np.savez_compressed(
        out_dir / "predictions.npz",
        frame_ids=np.asarray(frame_ids),
        image_paths=np.asarray(image_paths),
        gt_keypoints_2d_full=target,
        gt_joint_valid=valid,
        pred_keypoints_2d_prior_full=predicted["prior"],
        pred_keypoints_2d_query_full=predicted["query"],
        gt_keypoints_2d_crop=target_crop,
        pred_keypoints_2d_prior_crop=prediction_crop,
        tight_boxes=tight_box,
        tight_box_sizes=tight_size,
        prior_error_px=prior_summary["error"],
        coordinate_roundtrip_error_px=roundtrip_error,
    )
    for summary in summaries.values():
        summary.pop("error")
        summary.pop("frame_error")
    crop_offset = np.abs(prediction_crop - target_crop).max(axis=-1)
    valid_crop_offset = crop_offset[valid]
    search_coverage = {
        str(radius): float(np.mean(valid_crop_offset <= radius))
        for radius in (0.05, 0.10, 0.125, 0.20)
    }
    prior_metrics = summaries["prior"]
    local_median = prior_metrics[
        "similarity_aligned_bbox_normalized_error"
    ]["median"]
    local_p90 = prior_metrics[
        "similarity_aligned_bbox_normalized_error"
    ]["p90"]
    raw_median = prior_metrics["bbox_normalized_error"]["median"]
    if local_median is not None and (
        local_median > 0.04 or (local_p90 is not None and local_p90 > 0.08)
    ):
        recommendation = "train_joint_refinement"
    elif raw_median is not None and raw_median > 0.05:
        recommendation = "fix_global_camera_or_box_alignment_first"
    else:
        recommendation = "joint_refinement_is_low_priority"
    report = {
        "sequence_dir": str(sequence_dir),
        "checkpoint": str(checkpoint),
        "config": str(config),
        "mano_data_dir": (
            str(mano_data_dir) if mano_data_dir is not None else None
        ),
        "hand_side": hand_side,
        "box_source": "dexycb_gt_joints",
        "rescale_factor": args.rescale_factor,
        "frames": len(frame_ids),
        "valid_joints": int(valid.sum()),
        "coordinate_roundtrip_max_px": roundtrip_max,
        "refinement_search_radius_coverage": search_coverage,
        "recommendation": recommendation,
        "skipped_frames": skipped,
        "metrics": summaries,
        "overlay_files": overlays,
        "coordinate_contract": {
            "dexycb_gt": "original_image_pixels",
            "wilor_output": "crop_centered_normalized",
            "comparison": "original_image_pixels_after_inverse_affine",
            "left_hand": "mirrored_only_inside_wilor_crop_then_unmirrored",
        },
    }
    report_path = out_dir / "report.json"
    report_path.write_text(
        json.dumps(json_ready(report), indent=2), encoding="utf-8"
    )

    metric = report["metrics"]["prior"]
    print(f"frames: {report['frames']}")
    print(f"hand side: {hand_side}")
    print("pixel error:", metric["pixel_error"])
    print("bbox-normalized error:", metric["bbox_normalized_error"])
    print("wrist error:", metric["wrist_pixel_error"])
    print(
        "wrist-aligned error:", metric["wrist_aligned_pixel_error"]
    )
    print(
        "similarity-aligned error:",
        metric["similarity_aligned_pixel_error"],
    )
    print("PCK:", metric["pck"])
    ranked_joints = sorted(
        metric["per_joint_pixel_error"].items(),
        key=lambda item: (
            item[1]["median"] is not None,
            item[1]["median"] or float("-inf"),
        ),
        reverse=True,
    )
    print(
        "worst joints:",
        [
            (name, round(values["median"], 3))
            for name, values in ranked_joints[:5]
            if values["median"] is not None
        ],
    )
    print("crop search-radius coverage:", search_coverage)
    print("coordinate roundtrip max px:", roundtrip_max)
    print("recommendation:", recommendation)
    print("report:", report_path)
    print("overlays:", out_dir / "overlays")


if __name__ == "__main__":
    main()
