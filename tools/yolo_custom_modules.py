from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ECA(nn.Module):
    """Efficient Channel Attention with unchanged feature shape."""

    def __init__(self, k_size: int = 3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        return x * self.act(y)


class SE(nn.Module):
    """Squeeze-and-Excitation block with unchanged feature shape."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, channels, 1)
        self.act = nn.SiLU()
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)
        y = self.fc2(self.act(self.fc1(y)))
        return x * self.gate(y)


class FGDC(nn.Module):
    """Fine-Grained Detail Compensation block for YOLO neck features.

    The block preserves the input feature shape so it can be inserted in a
    YOLO neck YAML without changing downstream channel contracts.
    """

    def __init__(self, channels: int, reduction: int = 4, dilation: int = 2):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.local_detail = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
        )
        self.context_gate = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=dilation, dilation=dilation, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        detail = self.local_detail(x)
        gated_detail = detail * self.context_gate(x)
        return x + self.fuse(torch.cat((detail, gated_detail), dim=1))


class FocalBCEWithLogitsLoss(nn.Module):
    """Unreduced focal BCE, matching BCEWithLogitsLoss(reduction='none') shape."""

    def __init__(self, gamma: float = 1.5, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        prob = pred.sigmoid()
        pt = target * prob + (1.0 - target) * (1.0 - prob)
        loss = loss * (1.0 - pt).pow(self.gamma)
        if self.alpha > 0:
            alpha_factor = target * self.alpha + (1.0 - target) * (1.0 - self.alpha)
            loss = loss * alpha_factor
        return loss


def scale_aware_box_weight(
    target_bboxes: torch.Tensor,
    fg_mask: torch.Tensor,
    base_weight: torch.Tensor,
    imgsz: torch.Tensor,
    scale_weight: float = 1.0,
    tiny_area: float = 0.0005,
) -> torch.Tensor:
    """Increase regression weight for tiny boxes while leaving larger boxes unchanged."""
    if scale_weight <= 0:
        return base_weight
    selected = target_bboxes[fg_mask]
    widths = (selected[..., 2] - selected[..., 0]).clamp(min=0) / imgsz[1].clamp(min=1)
    heights = (selected[..., 3] - selected[..., 1]).clamp(min=0) / imgsz[0].clamp(min=1)
    areas = widths * heights
    tiny_factor = ((tiny_area - areas) / tiny_area).clamp(min=0.0, max=1.0).unsqueeze(-1)
    return base_weight * (1.0 + scale_weight * tiny_factor)


def defect_shape_aware_box_weight(
    target_bboxes: torch.Tensor,
    fg_mask: torch.Tensor,
    base_weight: torch.Tensor,
    imgsz: torch.Tensor,
    shape_weight: float = 1.0,
    tiny_area: float = 0.0005,
    elongated_ratio: float = 6.0,
    max_boost: float = 3.0,
) -> torch.Tensor:
    """Boost localization weight for tiny and elongated weld-defect boxes."""
    if shape_weight <= 0:
        return base_weight

    selected = target_bboxes[fg_mask]
    widths = (selected[..., 2] - selected[..., 0]).clamp(min=0) / imgsz[1].clamp(min=1)
    heights = (selected[..., 3] - selected[..., 1]).clamp(min=0) / imgsz[0].clamp(min=1)
    areas = widths * heights
    tiny_factor = ((tiny_area - areas) / tiny_area).clamp(min=0.0, max=1.0)

    short_side = torch.minimum(widths, heights).clamp(min=1e-6)
    long_side = torch.maximum(widths, heights)
    aspect = long_side / short_side
    elongated_factor = ((aspect - 1.0) / max(elongated_ratio - 1.0, 1e-6)).clamp(min=0.0, max=1.0)

    shape_factor = (tiny_factor + elongated_factor).clamp(min=0.0, max=float(max_boost))
    return base_weight * (1.0 + shape_weight * shape_factor.unsqueeze(-1))


def register_yolo_modules() -> None:
    """Register local YAML modules into Ultralytics' parse_model globals."""
    import ultralytics.nn.tasks as tasks

    tasks.ECA = ECA
    tasks.SE = SE
    tasks.FGDC = FGDC


def patch_detection_iou_loss(
    iou_type: str | None,
    scale_weight: float = 1.0,
    hard_bg_weight: float = 0.0,
    dsa_shape_weight: float = 1.0,
    dsa_tiny_area: float = 0.0005,
    dsa_elongated_ratio: float = 6.0,
    dsa_max_boost: float = 3.0,
) -> None:
    """Patch BboxLoss to use a selectable IoU variant for box regression."""
    if not iou_type or iou_type.lower() in {"ciou", "default"}:
        return

    from ultralytics.utils.loss import BboxLoss
    from ultralytics.utils.metrics import bbox_iou
    from ultralytics.utils.tal import bbox2dist

    selected = iou_type.lower()
    if selected not in {"diou", "giou", "iou", "sahb", "dsa"}:
        raise ValueError(f"Unsupported custom_iou_loss={iou_type}. Use ciou/default, diou, giou, iou, sahb, or dsa.")

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor,
        stride: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        if selected == "sahb":
            weight = scale_aware_box_weight(
                target_bboxes=target_bboxes,
                fg_mask=fg_mask,
                base_weight=weight,
                imgsz=imgsz,
                scale_weight=scale_weight,
            )
        elif selected == "dsa":
            weight = defect_shape_aware_box_weight(
                target_bboxes=target_bboxes,
                fg_mask=fg_mask,
                base_weight=weight,
                imgsz=imgsz,
                shape_weight=dsa_shape_weight,
                tiny_area=dsa_tiny_area,
                elongated_ratio=dsa_elongated_ratio,
                max_boost=dsa_max_boost,
            )
        kwargs = {
            "xywh": False,
            "CIoU": selected in {"ciou", "sahb", "dsa"},
            "DIoU": selected == "diou",
            "GIoU": selected == "giou",
        }
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], **kwargs)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            target_ltrb = bbox2dist(anchor_points, target_bboxes)
            target_ltrb = target_ltrb * stride
            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0]
            pred_scaled = pred_dist * stride
            pred_scaled[..., 0::2] /= imgsz[1]
            pred_scaled[..., 1::2] /= imgsz[0]
            loss_dfl = F.l1_loss(pred_scaled[fg_mask], target_ltrb[fg_mask], reduction="none").mean(-1, keepdim=True)
            loss_dfl = (loss_dfl * weight).sum() / target_scores_sum
        return loss_iou, loss_dfl

    BboxLoss.forward = forward


def patch_detection_cls_loss(cls_loss: str | None, gamma: float = 1.5, alpha: float = 0.25) -> None:
    """Patch detection loss to use unreduced focal BCE for classification."""
    if not cls_loss or cls_loss.lower() in {"bce", "default"}:
        return
    if cls_loss.lower() != "focal":
        raise ValueError(f"Unsupported custom_cls_loss={cls_loss}. Use bce/default or focal.")

    from ultralytics.utils.loss import v8DetectionLoss

    original_init = v8DetectionLoss.__init__

    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None):
        original_init(self, model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        self.bce = FocalBCEWithLogitsLoss(gamma=gamma, alpha=alpha)

    v8DetectionLoss.__init__ = __init__
