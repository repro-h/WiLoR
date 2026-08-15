#!/usr/bin/env python3
"""Export detector-based WiLoR query anchors for DexYCB streams.

The cache keeps WiLoR as a local hand estimator. Its weak-perspective camera is
used only to project MANO joints into the image; absolute camera translation is
intentionally not exported. Missing detections remain explicit invalid frames
so downstream temporal models can decide how to bridge them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from ultralytics import YOLO


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


CACHE_VERSION = "dexycb_wilor_query_cache_v1"
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
            "Run the official WiLoR detector and hand model over a sharded "
            "DexYCB hybrid manifest, then cache per-frame query anchors."
        )
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument(
        "--checkpoint", default="pretrained_models/wilor_final.ckpt"
    )
    parser.add_argument(
        "--config", default="pretrained_models/model_config.yaml"
    )
    parser.add_argument(
        "--detector", default="pretrained_models/detector.pt"
    )
    parser.add_argument(
        "--mano-data-dir",
        default=None,
        help="Directory containing MANO_RIGHT.pkl.",
    )
    parser.add_argument("--status-json", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--detector-confidence", type=float, default=0.3)
    parser.add_argument("--detector-batch-size", type=int, default=16)
    parser.add_argument("--model-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--rescale-factor", type=float, default=2.0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--stream-id",
        action="append",
        default=[],
        help="Export only this stream ID; may be supplied more than once.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--overlay-count",
        type=int,
        default=0,
        help="Save this many original-image query overlays per stream.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
    return rows


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_npz_atomic(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(path)


def color_paths(stream_dir: Path) -> list[Path]:
    paths = sorted(stream_dir.glob("color_*.jpg"))
    if not paths:
        paths = sorted(stream_dir.glob("color_*.png"))
    if not paths:
        raise FileNotFoundError(f"No color_*.jpg/png frames in {stream_dir}")
    return paths


def frame_id(path: Path) -> str:
    return path.stem.rsplit("_", 1)[-1].zfill(6)


def detector_side_classes(detector: YOLO) -> dict[str, int]:
    names = detector.names
    if isinstance(names, dict):
        normalized = {str(name).lower(): int(index) for index, name in names.items()}
    else:
        normalized = {str(name).lower(): index for index, name in enumerate(names)}
    mapping = {}
    for side in ("left", "right"):
        exact = [index for name, index in normalized.items() if name == side]
        partial = [index for name, index in normalized.items() if side in name]
        candidates = exact or partial
        if candidates:
            mapping[side] = int(candidates[0])
    if set(mapping) != {"left", "right"}:
        # This is the convention used by WiLoR's released two-class detector.
        if len(normalized) == 2:
            mapping = {"left": 0, "right": 1}
        else:
            raise ValueError(
                f"Cannot resolve left/right detector classes from {detector.names}"
            )
    return mapping


def select_detection(result: Any, expected_class: int) -> dict[str, Any]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return {
            "valid": False,
            "num_all": 0,
            "num_expected": 0,
        }
    xyxy = boxes.xyxy.detach().float().cpu().numpy().reshape(-1, 4)
    confidence = boxes.conf.detach().float().cpu().numpy().reshape(-1)
    classes = boxes.cls.detach().cpu().numpy().reshape(-1).astype(np.int64)
    expected = np.flatnonzero(classes == int(expected_class))
    candidates = expected if len(expected) else np.arange(len(classes))
    selected = int(candidates[np.argmax(confidence[candidates])])
    box = xyxy[selected].astype(np.float32)
    valid = (
        np.isfinite(box).all()
        and np.isfinite(confidence[selected])
        and box[2] > box[0]
        and box[3] > box[1]
    )
    return {
        "valid": bool(valid),
        "box": box,
        "confidence": float(confidence[selected]),
        "class_id": int(classes[selected]),
        "side_class_match": bool(classes[selected] == int(expected_class)),
        "num_all": int(len(classes)),
        "num_expected": int(len(expected)),
    }


def detect_frames(
    detector: YOLO,
    paths: list[Path],
    expected_class: int,
    confidence_threshold: float,
    batch_size: int,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    selected: list[dict[str, Any]] = []
    image_wh = np.zeros((len(paths), 2), dtype=np.int32)
    for start in range(0, len(paths), batch_size):
        chunk = paths[start : start + batch_size]
        images = []
        for offset, path in enumerate(chunk):
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(path)
            image_wh[start + offset] = [image.shape[1], image.shape[0]]
            images.append(image)
        results = detector.predict(
            source=images,
            conf=confidence_threshold,
            verbose=False,
        )
        if len(results) != len(chunk):
            raise RuntimeError(
                f"Detector returned {len(results)} results for {len(chunk)} frames"
            )
        selected.extend(
            select_detection(result, expected_class) for result in results
        )
    return selected, image_wh


class WiLoRQueryDataset(Dataset):
    def __init__(
        self,
        cfg: Any,
        records: list[dict[str, Any]],
        is_right: bool,
        rescale_factor: float,
    ) -> None:
        self.cfg = cfg
        self.records = records
        self.right = np.float32(is_right)
        self.rescale_factor = float(rescale_factor)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image = cv2.imread(str(record["path"]), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(record["path"])
        crop = ViTDetDataset(
            self.cfg,
            image,
            np.asarray(record["box"], dtype=np.float32)[None],
            np.asarray([self.right], dtype=np.float32),
            rescale_factor=self.rescale_factor,
        )[0]
        crop["frame_index"] = np.int64(record["frame_index"])
        return crop


def mirror_pixels(points: np.ndarray, image_wh: np.ndarray) -> np.ndarray:
    mirrored = points.copy()
    mirrored[..., 0] = image_wh[:, None, 0] - 1.0 - points[..., 0]
    return mirrored


def mirror_boxes(boxes: np.ndarray, image_wh: np.ndarray) -> np.ndarray:
    mirrored = boxes.copy()
    mirrored[:, 0] = image_wh[:, 0] - 1.0 - boxes[:, 2]
    mirrored[:, 2] = image_wh[:, 0] - 1.0 - boxes[:, 0]
    return mirrored


def distribution(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0, "median": None, "p90": None, "min": None, "max": None}
    return {
        "count": int(values.size),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def compact_status(summary: dict[str, Any], cached: bool = False) -> dict[str, Any]:
    return {
        "output": summary["output"],
        "frames": summary["frames"],
        "model_valid_frames": summary["model_valid_frames"],
        "cached": cached,
    }


def save_overlays(
    output_dir: Path,
    paths: list[Path],
    frame_ids: list[str],
    boxes: np.ndarray,
    joints: np.ndarray,
    valid: np.ndarray,
    count: int,
) -> list[str]:
    available = np.flatnonzero(valid)
    if count <= 0 or len(available) == 0:
        return []
    positions = np.linspace(0, len(available) - 1, min(count, len(available)))
    indices = available[np.round(positions).astype(np.int64)]
    overlay_dir = output_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for index in indices:
        image = cv2.imread(str(paths[index]), cv2.IMREAD_COLOR)
        if image is None:
            continue
        box = np.round(boxes[index]).astype(np.int32)
        cv2.rectangle(image, tuple(box[:2]), tuple(box[2:]), (0, 200, 255), 2)
        points = np.round(joints[index]).astype(np.int32)
        for parent, child in SKELETON_EDGES:
            cv2.line(
                image,
                tuple(points[parent]),
                tuple(points[child]),
                (30, 220, 30),
                2,
                cv2.LINE_AA,
            )
        for point in points:
            cv2.circle(image, tuple(point), 3, (20, 20, 240), -1, cv2.LINE_AA)
        destination = overlay_dir / f"{frame_ids[index]}.jpg"
        cv2.imwrite(str(destination), image)
        outputs.append(str(destination))
    return outputs


def export_stream(
    record: dict[str, Any],
    model: torch.nn.Module,
    cfg: Any,
    detector: YOLO,
    detector_classes: dict[str, int],
    device: torch.device,
    args: argparse.Namespace,
    stream_out: Path,
) -> dict[str, Any]:
    stream_id = str(record["stream_id"])
    hand_side = str(record["hand_side"]).lower()
    if hand_side not in {"left", "right"}:
        raise ValueError(f"Invalid hand_side={hand_side!r} for {stream_id}")
    stream_dir = Path(record["stream_dir"]).expanduser().resolve()
    if not stream_dir.is_dir():
        raise FileNotFoundError(stream_dir)
    paths = color_paths(stream_dir)
    frame_ids = [frame_id(path) for path in paths]
    detections, image_wh = detect_frames(
        detector,
        paths,
        detector_classes[hand_side],
        args.detector_confidence,
        args.detector_batch_size,
    )

    frames = len(paths)
    is_right = hand_side == "right"
    detection_valid = np.asarray(
        [value["valid"] for value in detections], dtype=bool
    )
    boxes = np.full((frames, 4), np.nan, dtype=np.float32)
    detector_confidence = np.full(frames, np.nan, dtype=np.float32)
    detector_class_id = np.full(frames, -1, dtype=np.int16)
    detector_side_class_match = np.zeros(frames, dtype=bool)
    num_all_candidates = np.asarray(
        [value["num_all"] for value in detections], dtype=np.int16
    )
    num_expected_candidates = np.asarray(
        [value["num_expected"] for value in detections], dtype=np.int16
    )
    for index in np.flatnonzero(detection_valid):
        boxes[index] = detections[index]["box"]
        detector_confidence[index] = detections[index]["confidence"]
        detector_class_id[index] = detections[index]["class_id"]
        detector_side_class_match[index] = detections[index]["side_class_match"]

    joints_uv_crop = np.full((frames, 21, 2), np.nan, dtype=np.float32)
    joints_uv_original = np.full((frames, 21, 2), np.nan, dtype=np.float32)
    joints_root_canonical = np.full((frames, 21, 3), np.nan, dtype=np.float32)
    joints_root_original = np.full((frames, 21, 3), np.nan, dtype=np.float32)
    vertices_root_canonical = np.full(
        (frames, 778, 3), np.nan, dtype=np.float32
    )
    vertices_root_original = np.full(
        (frames, 778, 3), np.nan, dtype=np.float32
    )
    crop_transform = np.full((frames, 2, 3), np.nan, dtype=np.float32)
    crop_box_center = np.full((frames, 2), np.nan, dtype=np.float32)
    crop_box_size = np.full(frames, np.nan, dtype=np.float32)
    model_valid = np.zeros(frames, dtype=bool)

    model_records = [
        {
            "frame_index": int(index),
            "path": paths[index],
            "box": boxes[index],
        }
        for index in np.flatnonzero(detection_valid)
    ]
    if model_records:
        dataset = WiLoRQueryDataset(
            cfg,
            model_records,
            is_right=is_right,
            rescale_factor=args.rescale_factor,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.model_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        with torch.inference_mode():
            for batch in loader:
                indices = batch.pop("frame_index").numpy().astype(np.int64)
                device_batch = recursive_to(batch, device)
                output = model(device_batch)
                query_key = (
                    "pred_keypoints_2d_query"
                    if "pred_keypoints_2d_query" in output
                    else "pred_keypoints_2d"
                )
                query_crop = output[query_key][:, :21]
                query_full = crop_normalized_to_full_image(
                    query_crop,
                    device_batch["crop_transform"],
                    cfg.MODEL.IMAGE_SIZE,
                    device_batch["img_size"][:, 0],
                    device_batch["right"],
                )
                joints_3d = output["pred_keypoints_3d"][:, :21]
                root = joints_3d[:, :1]
                joints_local = joints_3d - root
                vertices_local = output["pred_vertices"] - root
                multiplier = (2.0 * device_batch["right"] - 1.0).reshape(-1, 1)
                joints_original = joints_local.clone()
                joints_original[..., 0] *= multiplier
                vertices_original = vertices_local.clone()
                vertices_original[..., 0] *= multiplier

                chunks = {
                    "query_crop": query_crop.float().cpu().numpy(),
                    "query_full": query_full.float().cpu().numpy(),
                    "joints_local": joints_local.float().cpu().numpy(),
                    "joints_original": joints_original.float().cpu().numpy(),
                    "vertices_local": vertices_local.float().cpu().numpy(),
                    "vertices_original": vertices_original.float().cpu().numpy(),
                    "crop_transform": device_batch["crop_transform"].float().cpu().numpy(),
                    "box_center": device_batch["box_center"].float().cpu().numpy(),
                    "box_size": device_batch["box_size"].float().cpu().numpy(),
                }
                finite = np.ones(len(indices), dtype=bool)
                for value in chunks.values():
                    finite &= np.isfinite(value).reshape(len(indices), -1).all(axis=1)
                good_indices = indices[finite]
                if len(good_indices) == 0:
                    continue
                joints_uv_crop[good_indices] = chunks["query_crop"][finite]
                joints_uv_original[good_indices] = chunks["query_full"][finite]
                joints_root_canonical[good_indices] = chunks["joints_local"][finite]
                joints_root_original[good_indices] = chunks["joints_original"][finite]
                vertices_root_canonical[good_indices] = chunks["vertices_local"][finite]
                vertices_root_original[good_indices] = chunks["vertices_original"][finite]
                crop_transform[good_indices] = chunks["crop_transform"][finite]
                crop_box_center[good_indices] = chunks["box_center"][finite]
                crop_box_size[good_indices] = chunks["box_size"][finite]
                model_valid[good_indices] = True

    if is_right:
        joints_uv_canonical = joints_uv_original.copy()
        boxes_canonical = boxes.copy()
    else:
        joints_uv_canonical = mirror_pixels(joints_uv_original, image_wh)
        boxes_canonical = mirror_boxes(boxes, image_wh)
    uv_denominator = np.maximum(image_wh - 1, 1).astype(np.float32)
    joints_uv01_canonical = (
        joints_uv_canonical / uv_denominator[:, None]
    ).astype(np.float32)

    if model_valid.any():
        checked = np.flatnonzero(model_valid)
        reconstructed_crop = full_image_to_crop_normalized(
            torch.from_numpy(joints_uv_original[checked]),
            torch.from_numpy(crop_transform[checked]),
            cfg.MODEL.IMAGE_SIZE,
            torch.from_numpy(image_wh[checked, 0]),
            torch.full(
                (len(checked),),
                1.0 if is_right else 0.0,
                dtype=torch.float32,
            ),
        ).numpy()
        crop_roundtrip_max = float(np.max(np.abs(
            reconstructed_crop - joints_uv_crop[checked]
        )))
    else:
        crop_roundtrip_max = None
    if crop_roundtrip_max is not None and crop_roundtrip_max > 1e-3:
        raise RuntimeError(
            "WiLoR crop/full-image coordinate roundtrip exceeded 1e-3: "
            f"{crop_roundtrip_max:.6f}"
        )

    joint_in_frame_original = (
        np.isfinite(joints_uv_original).all(axis=-1)
        & (joints_uv_original[..., 0] >= 0)
        & (joints_uv_original[..., 0] < image_wh[:, None, 0])
        & (joints_uv_original[..., 1] >= 0)
        & (joints_uv_original[..., 1] < image_wh[:, None, 1])
    )
    joint_query_valid = model_valid[:, None] & joint_in_frame_original
    joint_query_confidence = np.where(
        joint_query_valid,
        detector_confidence[:, None],
        0.0,
    ).astype(np.float32)

    payload = {
        "frame_ids": np.asarray(frame_ids),
        "image_paths": np.asarray([str(path.resolve()) for path in paths]),
        "image_wh": image_wh,
        "bbox_xyxy_original": boxes,
        "bbox_xyxy_canonical_right": boxes_canonical,
        "detector_confidence": detector_confidence,
        "detector_class_id": detector_class_id,
        "detector_side_class_match": detector_side_class_match,
        "detector_num_candidates": num_all_candidates,
        "detector_num_expected_side_candidates": num_expected_candidates,
        "detection_valid": detection_valid,
        "model_valid": model_valid,
        "joints_uv_full_original": joints_uv_original,
        "joints_uv_full_canonical_right": joints_uv_canonical,
        "joints_uv01_canonical_right": joints_uv01_canonical,
        "joints_uv_crop_normalized_canonical_right": joints_uv_crop,
        "joint_in_frame_original": joint_in_frame_original,
        "joint_query_valid": joint_query_valid,
        "joint_query_confidence": joint_query_confidence,
        "joints_3d_root_relative_canonical_right": joints_root_canonical,
        "joints_3d_root_relative_original": joints_root_original,
        "vertices_3d_root_relative_canonical_right": vertices_root_canonical,
        "vertices_3d_root_relative_original": vertices_root_original,
        "mano_faces": np.asarray(model.mano.faces, dtype=np.int64),
        "crop_transform_canonical_right_to_crop": crop_transform,
        "crop_box_center_original": crop_box_center,
        "crop_box_size": crop_box_size,
        "joint_names": np.asarray(JOINT_NAMES),
        "stream_id": np.asarray(stream_id),
        "hand_side": np.asarray(hand_side),
        "mirrored_left_crop": np.asarray(not is_right),
        "canonical_right_horizontal_mirror": np.asarray(not is_right),
        "cache_version": np.asarray(CACHE_VERSION),
        "box_source": np.asarray("wilor_yolo_detector"),
        "query_projection_source": np.asarray("wilor_weak_perspective_2d_only"),
        "query_variant": np.asarray(
            "refined"
            if bool(getattr(model, "joint_refinement_enabled", False))
            else "mano_projected_prior"
        ),
        "joint_visibility_estimated": np.asarray(False),
        "joints_3d_unit": np.asarray("meters"),
        "vertices_3d_unit": np.asarray("meters"),
        "detector_confidence_threshold": np.float32(args.detector_confidence),
        "crop_rescale_factor": np.float32(args.rescale_factor),
        "absolute_camera_translation_exported": np.asarray(False),
        "ground_truth_used": np.asarray(False),
    }
    output_path = stream_out / "wilor_query_cache.npz"
    write_npz_atomic(output_path, payload)
    overlays = save_overlays(
        stream_out,
        paths,
        frame_ids,
        boxes,
        joints_uv_original,
        model_valid,
        args.overlay_count,
    )
    summary = {
        "cache_version": CACHE_VERSION,
        "stream_id": stream_id,
        "hand_side": hand_side,
        "stream_dir": str(stream_dir),
        "output": str(output_path),
        "frames": frames,
        "detected_frames": int(detection_valid.sum()),
        "model_valid_frames": int(model_valid.sum()),
        "detection_fraction": float(detection_valid.mean()),
        "model_valid_fraction": float(model_valid.mean()),
        "joint_query_valid_fraction": float(joint_query_valid.mean()),
        "detector_side_class_match_fraction": (
            float(detector_side_class_match[detection_valid].mean())
            if detection_valid.any()
            else None
        ),
        "detector_confidence": distribution(detector_confidence[model_valid]),
        "detector_confidence_threshold": args.detector_confidence,
        "crop_rescale_factor": args.rescale_factor,
        "crop_coordinate_roundtrip_max": crop_roundtrip_max,
        "coordinate_frames": {
            "joints_uv_full_original": "original_camera_image_pixels",
            "joints_uv_full_canonical_right": (
                "original_camera_image_pixels"
                if is_right
                else "horizontally_mirrored_image_pixels"
            ),
            "joints_uv01_canonical_right": "canonical_right_image_uv_in_0_1",
            "joints_3d_root_relative_original": "original_camera_axes_wrist_origin",
            "joints_3d_root_relative_canonical_right": "right_hand_mano_wrist_origin",
        },
        "ground_truth_used": False,
        "hand_side_source": "hybrid_manifest_metadata",
        "query_variant": (
            "refined"
            if bool(getattr(model, "joint_refinement_enabled", False))
            else "mano_projected_prior"
        ),
        "joint_visibility_estimated": False,
        "absolute_camera_translation_exported": False,
        "overlays": overlays,
    }
    write_json_atomic(stream_out / "summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.detector_confidence <= 1.0:
        raise ValueError("--detector-confidence must be in [0, 1]")
    for name in ("detector_batch_size", "model_batch_size", "num_workers"):
        value = getattr(args, name)
        if value <= 0 and name != "num_workers":
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
        if name == "num_workers" and value < 0:
            raise ValueError("--num-workers cannot be negative")
    if args.rescale_factor <= 0:
        raise ValueError("--rescale-factor must be positive")
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, num-shards)")

    manifest = resolve_path(args.manifest)
    checkpoint = resolve_path(args.checkpoint)
    config = resolve_path(args.config)
    detector_path = resolve_path(args.detector)
    out_root = resolve_path(args.out_root)
    mano_data_dir = (
        resolve_path(args.mano_data_dir) if args.mano_data_dir else None
    )
    status_path = (
        resolve_path(args.status_json)
        if args.status_json
        else out_root / f"status_shard_{args.shard_index}.json"
    )
    for path in (manifest, checkpoint, config, detector_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if mano_data_dir is not None and not (
        mano_data_dir / "MANO_RIGHT.pkl"
    ).is_file():
        raise FileNotFoundError(mano_data_dir / "MANO_RIGHT.pkl")

    records = load_jsonl(manifest)
    stream_ids = [str(record.get("stream_id", "")) for record in records]
    if any(not value for value in stream_ids):
        raise KeyError("Every manifest row must contain stream_id")
    if len(stream_ids) != len(set(stream_ids)):
        raise ValueError("Manifest contains duplicate stream_id rows")
    if args.stream_id:
        requested = set(args.stream_id)
        available = set(stream_ids)
        missing = sorted(requested - available)
        if missing:
            raise KeyError(f"Requested streams are absent from manifest: {missing}")
        records = [
            record for record in records if str(record["stream_id"]) in requested
        ]
    records = records[args.shard_index :: args.num_shards]
    if args.limit > 0:
        records = records[: args.limit]
    out_root.mkdir(parents=True, exist_ok=True)

    os.chdir(REPO_ROOT)
    device = torch.device(args.device)
    model, cfg = load_wilor(
        str(checkpoint),
        str(config),
        init_renderer=False,
        mano_data_dir=mano_data_dir,
    )
    model = model.to(device).eval()
    detector = YOLO(str(detector_path))
    detector = detector.to(device)
    detector_classes = detector_side_classes(detector)
    state = {
        "cache_version": CACHE_VERSION,
        "manifest": str(manifest),
        "out_root": str(out_root),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "num_requested": len(records),
        "detector_classes": detector_classes,
        "completed": {},
        "failed": {},
    }

    for index, record in enumerate(records, start=1):
        stream_id = str(record["stream_id"])
        stream_out = out_root / stream_id
        output_path = stream_out / "wilor_query_cache.npz"
        summary_path = stream_out / "summary.json"
        if output_path.is_file() and summary_path.is_file() and not args.overwrite:
            print(f"[{index}/{len(records)}] {stream_id}: cached", flush=True)
            state["completed"][stream_id] = compact_status(
                json.loads(summary_path.read_text(encoding="utf-8")),
                cached=True,
            )
            write_json_atomic(status_path, state)
            continue
        stream_out.mkdir(parents=True, exist_ok=True)
        print(
            f"[{index}/{len(records)}] {stream_id} "
            f"side={record.get('hand_side')}",
            flush=True,
        )
        try:
            summary = export_stream(
                record,
                model,
                cfg,
                detector,
                detector_classes,
                device,
                args,
                stream_out,
            )
            state["completed"][stream_id] = compact_status(summary)
            error_log = stream_out / "error.log"
            if error_log.is_file():
                error_log.unlink()
            print(
                f"  done: {summary['model_valid_frames']}/{summary['frames']} "
                "valid frames",
                flush=True,
            )
        except Exception as error:
            state["failed"][stream_id] = repr(error)
            (stream_out / "error.log").write_text(
                traceback.format_exc(), encoding="utf-8"
            )
            print(f"  FAILED: {error!r}", flush=True)
        write_json_atomic(status_path, state)

    state["num_completed"] = len(state["completed"])
    state["num_failed"] = len(state["failed"])
    write_json_atomic(status_path, state)
    print(json.dumps({
        "cache_version": CACHE_VERSION,
        "manifest": str(manifest),
        "out_root": str(out_root),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "num_requested": len(records),
        "num_completed": state["num_completed"],
        "num_failed": state["num_failed"],
        "status_json": str(status_path),
    }, indent=2), flush=True)
    if state["failed"]:
        raise RuntimeError(
            f"{len(state['failed'])} WiLoR query exports failed; see {status_path}"
        )


if __name__ == "__main__":
    main()
