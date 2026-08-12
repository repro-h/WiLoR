import torch
import torch.nn as nn
import torch.nn.functional as F

class Keypoint2DLoss(nn.Module):

    def __init__(self, loss_type: str = 'l1'):
        """
        2D keypoint loss module.
        Args:
            loss_type (str): Choose between l1 and l2 losses.
        """
        super(Keypoint2DLoss, self).__init__()
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError('Unsupported loss function')

    def forward(self, pred_keypoints_2d: torch.Tensor, gt_keypoints_2d: torch.Tensor) -> torch.Tensor:
        """
        Compute 2D reprojection loss on the keypoints.
        Args:
            pred_keypoints_2d (torch.Tensor): Tensor of shape [B, S, N, 2] containing projected 2D keypoints (B: batch_size, S: num_samples, N: num_keypoints)
            gt_keypoints_2d (torch.Tensor): Tensor of shape [B, S, N, 3] containing the ground truth 2D keypoints and confidence.
        Returns:
            torch.Tensor: 2D keypoint loss.
        """
        if gt_keypoints_2d.shape[-1] < 2:
            raise ValueError("gt_keypoints_2d must contain x and y")
        target = gt_keypoints_2d[:, :, :2]
        if gt_keypoints_2d.shape[-1] >= 3:
            conf = gt_keypoints_2d[:, :, 2:3].clamp_min(0.0)
        else:
            conf = torch.ones_like(target[:, :, :1])
        finite = torch.isfinite(target).all(dim=-1, keepdim=True)
        conf = torch.where(finite, conf, torch.zeros_like(conf))
        target = torch.nan_to_num(target)
        loss = (conf * self.loss_fn(pred_keypoints_2d, target)).sum(dim=(1,2))
        return loss.sum()


class Keypoint3DLoss(nn.Module):

    def __init__(self, loss_type: str = 'l1'):
        """
        3D keypoint loss module.
        Args:
            loss_type (str): Choose between l1 and l2 losses.
        """
        super(Keypoint3DLoss, self).__init__()
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError('Unsupported loss function')

    def forward(self, pred_keypoints_3d: torch.Tensor, gt_keypoints_3d: torch.Tensor, pelvis_id: int = 0):
        """
        Compute 3D keypoint loss.
        Args:
            pred_keypoints_3d (torch.Tensor): Tensor of shape [B, S, N, 3] containing the predicted 3D keypoints (B: batch_size, S: num_samples, N: num_keypoints)
            gt_keypoints_3d (torch.Tensor): Tensor of shape [B, S, N, 4] containing the ground truth 3D keypoints and confidence.
        Returns:
            torch.Tensor: 3D keypoint loss.
        """
        batch_size = pred_keypoints_3d.shape[0]
        gt_keypoints_3d = gt_keypoints_3d.clone()
        pred_keypoints_3d = pred_keypoints_3d - pred_keypoints_3d[:, pelvis_id, :].unsqueeze(dim=1)
        gt_keypoints_3d[:, :, :-1] = gt_keypoints_3d[:, :, :-1] - gt_keypoints_3d[:, pelvis_id, :-1].unsqueeze(dim=1)
        conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
        gt_keypoints_3d = gt_keypoints_3d[:, :, :-1]
        loss = (conf * self.loss_fn(pred_keypoints_3d, gt_keypoints_3d)).sum(dim=(1,2))
        return loss.sum()

class ParameterLoss(nn.Module):

    def __init__(self):
        """
        MANO parameter loss module.
        """
        super(ParameterLoss, self).__init__()
        self.loss_fn = nn.MSELoss(reduction='none')

    def forward(self, pred_param: torch.Tensor, gt_param: torch.Tensor, has_param: torch.Tensor):
        """
        Compute MANO parameter loss.
        Args:
            pred_param (torch.Tensor): Tensor of shape [B, S, ...] containing the predicted parameters (body pose / global orientation / betas)
            gt_param (torch.Tensor): Tensor of shape [B, S, ...] containing the ground truth MANO parameters.
        Returns:
            torch.Tensor: L2 parameter loss loss.
        """
        batch_size = pred_param.shape[0]
        num_dims = len(pred_param.shape)
        mask_dimension = [batch_size] + [1] * (num_dims-1)
        has_param = has_param.type(pred_param.type()).view(*mask_dimension)
        loss_param = (has_param * self.loss_fn(pred_param, gt_param))
        return loss_param.sum()


class JointRefinementLoss(nn.Module):
    """Supervise local 2D refinement without assigning visibility semantics.

    The last value in ``gt_keypoints_2d`` is treated only as an annotation
    weight. It may represent validity, confidence, or visibility depending on
    the source dataset; this loss deliberately does not reinterpret it.
    """

    def __init__(
        self,
        smooth_l1_beta: float = 0.01,
        heatmap_sigma: float = 0.025,
        uv_weight: float = 1.0,
        heatmap_weight: float = 1.0,
        delta_weight: float = 0.01,
    ) -> None:
        super().__init__()
        if smooth_l1_beta <= 0:
            raise ValueError("smooth_l1_beta must be positive")
        if heatmap_sigma <= 0:
            raise ValueError("heatmap_sigma must be positive")
        self.smooth_l1_beta = float(smooth_l1_beta)
        self.heatmap_sigma = float(heatmap_sigma)
        self.uv_weight = float(uv_weight)
        self.heatmap_weight = float(heatmap_weight)
        self.delta_weight = float(delta_weight)

    @staticmethod
    def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        weights = weights.to(values.dtype)
        return (values * weights).sum() / weights.sum().clamp_min(1.0)

    def forward(
        self,
        prediction: dict[str, torch.Tensor],
        gt_keypoints_2d: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        refined_uv = prediction["refined_uv"]
        prior_uv = prediction["prior_uv"]
        candidate_uv = prediction["candidate_uv"]
        logits = prediction["heatmap_logits"]

        num_joints = refined_uv.shape[1]
        if gt_keypoints_2d.shape[-1] < 2:
            raise ValueError("gt_keypoints_2d must contain x and y")
        gt_uv = gt_keypoints_2d[:, :num_joints, :2]
        if gt_keypoints_2d.shape[-1] >= 3:
            annotation_weight = gt_keypoints_2d[
                :, :num_joints, 2
            ].clamp_min(0.0)
        else:
            annotation_weight = torch.ones_like(gt_uv[..., 0])
        finite = torch.isfinite(gt_uv).all(dim=-1)
        annotation_weight = torch.where(
            finite, annotation_weight, torch.zeros_like(annotation_weight)
        )
        gt_uv = torch.nan_to_num(gt_uv)

        uv_error = F.smooth_l1_loss(
            refined_uv,
            gt_uv,
            reduction="none",
            beta=self.smooth_l1_beta,
        ).sum(dim=-1)
        uv_loss = self._weighted_mean(uv_error, annotation_weight)

        squared_distance = (candidate_uv - gt_uv[:, :, None]).square().sum(dim=-1)
        target_logits = -squared_distance / (2.0 * self.heatmap_sigma ** 2)
        target_probability = F.softmax(target_logits, dim=-1)
        heatmap_error = -(
            target_probability * F.log_softmax(logits, dim=-1)
        ).sum(dim=-1)

        search_radius = (
            candidate_uv - prior_uv[:, :, None]
        ).abs().amax(dim=(2, 3))
        in_search_window = (
            (gt_uv - prior_uv).abs().amax(dim=-1) <= search_radius
        )
        heatmap_weight = annotation_weight * in_search_window.to(annotation_weight.dtype)
        heatmap_loss = self._weighted_mean(heatmap_error, heatmap_weight)

        delta_error = prediction["delta_uv"].abs().sum(dim=-1)
        delta_loss = self._weighted_mean(delta_error, annotation_weight)
        total = (
            self.uv_weight * uv_loss
            + self.heatmap_weight * heatmap_loss
            + self.delta_weight * delta_loss
        )
        valid_fraction = (annotation_weight > 0).to(refined_uv.dtype).mean()
        in_window_fraction = self._weighted_mean(
            in_search_window.to(refined_uv.dtype),
            (annotation_weight > 0).to(refined_uv.dtype),
        )
        return {
            "loss": total,
            "uv": uv_loss,
            "heatmap": heatmap_loss,
            "delta": delta_loss,
            "valid_fraction": valid_fraction,
            "in_search_window_fraction": in_window_fraction,
        }
