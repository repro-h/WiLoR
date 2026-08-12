"""Image-conditioned refinement of MANO-projected 2D hand joints."""

from __future__ import annotations

from math import gcd
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class JointRefinementHead(nn.Module):
    """Refine projected MANO joints with local evidence from the ViT feature map.

    ``prior_uv`` follows WiLoR's crop-centered convention: ``(-0.5, -0.5)``
    is the top-left of the 256x256 crop and ``(0.5, 0.5)`` is the bottom-right.
    The ViT receives the centered 256x192 slice, so x coordinates are remapped
    before sampling its spatial feature map.
    """

    def __init__(
        self,
        feature_dim: int = 1280,
        hidden_dim: int = 256,
        num_joints: int = 21,
        window_size: int = 9,
        search_radius: float = 0.125,
        crop_size: int = 256,
        backbone_width: int = 192,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if window_size < 3 or window_size % 2 == 0:
            raise ValueError("window_size must be an odd integer >= 3")
        if search_radius <= 0:
            raise ValueError("search_radius must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self.num_joints = int(num_joints)
        self.window_size = int(window_size)
        self.search_radius = float(search_radius)
        self.crop_size = int(crop_size)
        self.backbone_width = int(backbone_width)
        self.temperature = float(temperature)

        normalization_groups = gcd(32, hidden_dim)
        self.feature_projection = nn.Sequential(
            nn.Conv2d(feature_dim, hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(normalization_groups, hidden_dim),
            nn.GELU(),
        )
        self.joint_embedding = nn.Embedding(num_joints, hidden_dim)
        self.score_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        # Uniform probabilities have exactly zero expected offset on this
        # symmetric grid. A newly initialized head therefore preserves WiLoR.
        nn.init.zeros_(self.score_head[-1].weight)
        nn.init.zeros_(self.score_head[-1].bias)

        axis = torch.linspace(-search_radius, search_radius, window_size)
        offset_y, offset_x = torch.meshgrid(axis, axis, indexing="ij")
        offsets = torch.stack((offset_x, offset_y), dim=-1).reshape(-1, 2)
        self.register_buffer("candidate_offsets", offsets, persistent=False)

    def crop_uv_to_feature_grid(self, crop_uv: torch.Tensor) -> torch.Tensor:
        """Convert centered crop UV to ``grid_sample`` coordinates."""
        horizontal_scale = self.crop_size / self.backbone_width
        x = 2.0 * horizontal_scale * crop_uv[..., 0]
        y = 2.0 * crop_uv[..., 1]
        return torch.stack((x, y), dim=-1)

    def forward(
        self,
        image_features: torch.Tensor,
        prior_uv: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if image_features.ndim != 4:
            raise ValueError(
                "image_features must have shape [B, C, H, W], got "
                f"{tuple(image_features.shape)}"
            )
        if prior_uv.ndim != 3 or prior_uv.shape[-1] != 2:
            raise ValueError(
                f"prior_uv must have shape [B, J, 2], got {tuple(prior_uv.shape)}"
            )
        if prior_uv.shape[1] != self.num_joints:
            raise ValueError(
                f"Expected {self.num_joints} joints, got {prior_uv.shape[1]}"
            )

        batch_size, num_joints = prior_uv.shape[:2]
        num_candidates = self.candidate_offsets.shape[0]
        candidate_uv = (
            prior_uv[:, :, None, :]
            + self.candidate_offsets.view(1, 1, num_candidates, 2)
        )
        sample_grid = self.crop_uv_to_feature_grid(candidate_uv)

        features = self.feature_projection(image_features)
        sampled = F.grid_sample(
            features,
            sample_grid.reshape(batch_size, num_joints * num_candidates, 1, 2),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampled = sampled[:, :, :, 0].transpose(1, 2).reshape(
            batch_size, num_joints, num_candidates, -1
        )

        joint_ids = torch.arange(num_joints, device=prior_uv.device)
        sampled = sampled + self.joint_embedding(joint_ids)[None, :, None, :]
        logits = self.score_head(sampled).squeeze(-1)
        probabilities = F.softmax(logits / self.temperature, dim=-1)
        delta_uv = torch.sum(
            probabilities[..., None]
            * self.candidate_offsets.view(1, 1, num_candidates, 2),
            dim=2,
        )
        refined_uv = prior_uv + delta_uv

        entropy = -torch.sum(
            probabilities * probabilities.clamp_min(1e-8).log(), dim=-1
        )
        max_entropy = probabilities.new_tensor(float(num_candidates)).log()
        localization_confidence = (1.0 - entropy / max_entropy).clamp(0.0, 1.0)
        prior_in_feature = (
            self.crop_uv_to_feature_grid(prior_uv).abs() <= 1.0
        ).all(dim=-1)

        return {
            "prior_uv": prior_uv,
            "candidate_uv": candidate_uv,
            "heatmap_logits": logits,
            "heatmap_probabilities": probabilities,
            "delta_uv": delta_uv,
            "refined_uv": refined_uv,
            "localization_confidence": localization_confidence,
            "prior_in_feature": prior_in_feature,
        }
