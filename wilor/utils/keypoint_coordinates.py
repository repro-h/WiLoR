"""Coordinate transforms for WiLoR's cropped hand images."""

from __future__ import annotations

from collections.abc import Mapping

import torch


def _require_batched(
    points: torch.Tensor,
    crop_transform: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if points.ndim < 3 or points.shape[-1] < 2:
        raise ValueError(
            "points must have shape [B, ..., 2+] with a batch dimension, got "
            f"{tuple(points.shape)}"
        )
    if crop_transform.ndim != 3 or crop_transform.shape[-2:] != (2, 3):
        raise ValueError(
            "crop_transform must have shape [B, 2, 3], got "
            f"{tuple(crop_transform.shape)}"
        )
    if points.shape[0] != crop_transform.shape[0]:
        raise ValueError("points and crop_transform batch sizes differ")
    return points, crop_transform.to(device=points.device, dtype=points.dtype)


def _batch_vector(
    value: torch.Tensor,
    points: torch.Tensor,
    name: str,
) -> torch.Tensor:
    value = torch.as_tensor(value, device=points.device, dtype=points.dtype)
    value = value.reshape(points.shape[0], -1)
    if value.shape[1] != 1:
        raise ValueError(f"{name} must contain one value per batch item")
    return value


def full_image_to_crop_normalized(
    points: torch.Tensor,
    crop_transform: torch.Tensor,
    crop_size: int | float,
    image_width: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    """Map full-image pixels to WiLoR's centered crop coordinates.

    ``crop_transform`` is the affine matrix returned by
    :func:`generate_image_patch_cv2`. Left-hand inputs are mirrored before
    that affine transform, matching :class:`ViTDetDataset` exactly. Channels
    after x and y (for example an annotation weight) are preserved.
    """
    points, crop_transform = _require_batched(points, crop_transform)
    if crop_size <= 0:
        raise ValueError("crop_size must be positive")

    original_shape = points.shape
    flat = points[..., :2].reshape(points.shape[0], -1, 2)
    width = _batch_vector(image_width, points, "image_width")
    handedness = _batch_vector(right, points, "right")

    transformed = flat.clone()
    transformed[..., 0] = torch.where(
        handedness >= 0.5,
        flat[..., 0],
        width - 1.0 - flat[..., 0],
    )
    homogeneous = torch.cat(
        (transformed, torch.ones_like(transformed[..., :1])), dim=-1
    )
    crop_pixels = torch.einsum(
        "bij,bnj->bni", crop_transform, homogeneous
    )
    crop_uv = crop_pixels / float(crop_size) - 0.5
    crop_uv = crop_uv.reshape(*original_shape[:-1], 2)
    if original_shape[-1] == 2:
        return crop_uv
    return torch.cat((crop_uv, points[..., 2:]), dim=-1)


def crop_normalized_to_full_image(
    points: torch.Tensor,
    crop_transform: torch.Tensor,
    crop_size: int | float,
    image_width: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    """Invert :func:`full_image_to_crop_normalized`."""
    points, crop_transform = _require_batched(points, crop_transform)
    if crop_size <= 0:
        raise ValueError("crop_size must be positive")

    original_shape = points.shape
    flat = points[..., :2].reshape(points.shape[0], -1, 2)
    crop_pixels = (flat + 0.5) * float(crop_size)
    homogeneous = torch.cat(
        (crop_pixels, torch.ones_like(crop_pixels[..., :1])), dim=-1
    )

    affine = torch.eye(
        3, device=points.device, dtype=points.dtype
    )[None].repeat(points.shape[0], 1, 1)
    affine[:, :2] = crop_transform
    inverse = torch.linalg.inv(affine)
    flipped_image = torch.einsum(
        "bij,bnj->bni", inverse, homogeneous
    )[..., :2]

    width = _batch_vector(image_width, points, "image_width")
    handedness = _batch_vector(right, points, "right")
    full_image = flipped_image.clone()
    full_image[..., 0] = torch.where(
        handedness >= 0.5,
        flipped_image[..., 0],
        width - 1.0 - flipped_image[..., 0],
    )
    full_image = full_image.reshape(*original_shape[:-1], 2)
    if original_shape[-1] == 2:
        return full_image
    return torch.cat((full_image, points[..., 2:]), dim=-1)


def keypoints_2d_crop_from_batch(
    batch: Mapping[str, torch.Tensor],
    crop_size: int | float,
) -> torch.Tensor:
    """Resolve 2D supervision in crop coordinates.

    New datasets should expose ``keypoints_2d_crop`` explicitly. The legacy
    WiLoR ``keypoints_2d`` key remains supported because its loaders already
    convert full-image annotations in ``get_example``. A loader may instead
    provide ``keypoints_2d_full`` (or ``orig_keypoints_2d``) together with the
    affine crop metadata, in which case the conversion happens here.
    """
    if "keypoints_2d_crop" in batch:
        return batch["keypoints_2d_crop"]
    if "keypoints_2d" in batch:
        keypoints = batch["keypoints_2d"]
        finite_xy = keypoints[..., :2][torch.isfinite(keypoints[..., :2])]
        if finite_xy.numel() and finite_xy.abs().max().item() > 8.0:
            raise ValueError(
                "Legacy keypoints_2d appears to contain full-image pixels. "
                "Expose it as keypoints_2d_full with crop_transform, img_size, "
                "and right so it can be transformed before loss computation."
            )
        return keypoints

    full_key = next(
        (
            key
            for key in ("keypoints_2d_full", "orig_keypoints_2d")
            if key in batch
        ),
        None,
    )
    if full_key is None:
        raise KeyError(
            "Batch lacks keypoints_2d_crop, legacy keypoints_2d, and "
            "full-image 2D keypoints"
        )
    transform_key = next(
        (key for key in ("crop_transform", "trans") if key in batch), None
    )
    if transform_key is None:
        raise KeyError("Full-image 2D keypoints require crop_transform")
    if "right" not in batch:
        raise KeyError("Full-image 2D keypoints require right handedness")

    if "img_size" in batch:
        image_width = batch["img_size"][..., 0]
    elif "image_width" in batch:
        image_width = batch["image_width"]
    else:
        raise KeyError("Full-image 2D keypoints require img_size or image_width")

    return full_image_to_crop_normalized(
        batch[full_key],
        batch[transform_key],
        crop_size,
        image_width,
        batch["right"],
    )
