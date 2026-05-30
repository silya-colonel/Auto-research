from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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


def load_hnc_sidecar(sidecar_path: str | Path | None) -> dict[str, list[dict[str, Any]]]:
    """Load mined false-positive boxes keyed by train image basename and stem."""
    if not sidecar_path:
        return {}
    path = Path(sidecar_path)
    if not path.exists():
        raise FileNotFoundError(f"HNC sidecar not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_entries = payload.get("entries", payload)
    if not isinstance(raw_entries, dict):
        raise ValueError(f"HNC sidecar entries must be a mapping: {path}")

    entries: dict[str, list[dict[str, Any]]] = {}
    for key, detections in raw_entries.items():
        if not isinstance(detections, list):
            continue
        clean: list[dict[str, Any]] = []
        for det in detections:
            if not isinstance(det, dict):
                continue
            box = det.get("xyxyn") or det.get("box") or det.get("bbox")
            cls = det.get("cls", det.get("class"))
            conf = det.get("conf", det.get("confidence", 1.0))
            if box is None or cls is None or len(box) != 4:
                continue
            clean.append({"xyxyn": [float(v) for v in box], "cls": int(cls), "conf": float(conf)})
        if clean:
            name = Path(str(key)).name
            entries[name] = clean
            entries[Path(name).stem] = clean
    return entries


def hnc_region_weights_from_batch(
    batch: dict[str, Any],
    anchor_points: torch.Tensor,
    stride_tensor: torch.Tensor,
    pred_scores: torch.Tensor,
    sidecar: dict[str, list[dict[str, Any]]],
    imgsz: torch.Tensor,
    high_conf: float = 0.75,
    mid_weight: float = 0.3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return image mask and per-logit weights for mined FP regions.

    Region mapping is intentionally local: an anchor is penalized only when its
    anchor point falls inside a mined FP box, and only for that mined predicted
    class. This avoids reverting to whole-image background suppression.
    """
    hnc_images = torch.zeros(pred_scores.shape[0], dtype=torch.bool, device=pred_scores.device)
    weights = torch.zeros_like(pred_scores)
    im_files = batch.get("im_file") or []
    if not sidecar or not im_files:
        return hnc_images, weights

    anchor_xy = anchor_points * stride_tensor
    img_h = float(imgsz[0].detach().cpu())
    img_w = float(imgsz[1].detach().cpu())
    num_classes = pred_scores.shape[-1]
    for batch_idx, im_file in enumerate(im_files):
        name = Path(str(im_file)).name
        detections = sidecar.get(name) or sidecar.get(Path(name).stem)
        if not detections:
            continue
        hnc_images[batch_idx] = True
        for det in detections:
            cls = int(det["cls"])
            if cls < 0 or cls >= num_classes:
                continue
            conf = float(det.get("conf", 1.0))
            region_weight = 1.0 if conf >= high_conf else mid_weight
            if region_weight <= 0:
                continue
            x1, y1, x2, y2 = det["xyxyn"]
            box = torch.tensor([x1 * img_w, y1 * img_h, x2 * img_w, y2 * img_h], device=pred_scores.device)
            inside = (
                (anchor_xy[:, 0] >= box[0])
                & (anchor_xy[:, 0] <= box[2])
                & (anchor_xy[:, 1] >= box[1])
                & (anchor_xy[:, 1] <= box[3])
            )
            weights[batch_idx, inside, cls] = torch.maximum(
                weights[batch_idx, inside, cls],
                torch.tensor(region_weight, device=pred_scores.device, dtype=weights.dtype),
            )
    return hnc_images, weights


def xywhn_to_xyxyn(boxes: torch.Tensor) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.reshape(0, 4)
    cx, cy, bw, bh = boxes.unbind(-1)
    return torch.stack((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2), dim=-1)


def single_box_iou(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.zeros(0, device=box.device, dtype=box.dtype)
    ix1 = torch.maximum(box[0], boxes[:, 0])
    iy1 = torch.maximum(box[1], boxes[:, 1])
    ix2 = torch.minimum(box[2], boxes[:, 2])
    iy2 = torch.minimum(box[3], boxes[:, 3])
    inter = (ix2 - ix1).clamp_min(0) * (iy2 - iy1).clamp_min(0)
    area1 = ((box[2] - box[0]).clamp_min(0) * (box[3] - box[1]).clamp_min(0)).clamp_min(1e-9)
    area2 = (boxes[:, 2] - boxes[:, 0]).clamp_min(0) * (boxes[:, 3] - boxes[:, 1]).clamp_min(0)
    return inter / (area1 + area2 - inter).clamp_min(1e-9)


def hnc_adaptive_weights_from_batch(
    batch: dict[str, Any],
    anchor_points: torch.Tensor,
    stride_tensor: torch.Tensor,
    pred_scores: torch.Tensor,
    sidecar: dict[str, list[dict[str, Any]]],
    imgsz: torch.Tensor,
    iou_ref: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return entropy-IoU adaptive weights for HNC-v2 mined FP regions."""
    hnc_images = torch.zeros(pred_scores.shape[0], dtype=torch.bool, device=pred_scores.device)
    weights = torch.zeros_like(pred_scores)
    im_files = batch.get("im_file") or []
    if not sidecar or not im_files:
        return hnc_images, weights

    anchor_xy = anchor_points * stride_tensor
    img_h = float(imgsz[0].detach().cpu())
    img_w = float(imgsz[1].detach().cpu())
    num_classes = pred_scores.shape[-1]
    eps = torch.finfo(pred_scores.dtype).eps

    if num_classes > 1:
        probs = pred_scores.softmax(dim=-1).clamp_min(eps)
        entropy = -(probs * probs.log()).sum(dim=-1) / torch.log(
            torch.tensor(float(num_classes), device=pred_scores.device, dtype=pred_scores.dtype)
        )
        certainty = (1.0 - entropy).clamp(0.0, 1.0).detach()
    else:
        prob = pred_scores.sigmoid().clamp(eps, 1.0 - eps)
        entropy = -(prob * prob.log() + (1.0 - prob) * (1.0 - prob).log()) / torch.log(
            torch.tensor(2.0, device=pred_scores.device, dtype=pred_scores.dtype)
        )
        certainty = (1.0 - entropy).clamp(0.0, 1.0).detach()

    gt_batch_idx = batch.get("batch_idx", torch.empty(0, device=pred_scores.device))
    gt_bboxes = batch.get("bboxes", torch.empty(0, 4, device=pred_scores.device))
    if not torch.is_tensor(gt_batch_idx):
        gt_batch_idx = torch.as_tensor(gt_batch_idx, device=pred_scores.device)
    if not torch.is_tensor(gt_bboxes):
        gt_bboxes = torch.as_tensor(gt_bboxes, device=pred_scores.device, dtype=pred_scores.dtype)
    gt_batch_idx = gt_batch_idx.to(device=pred_scores.device)
    gt_bboxes = gt_bboxes.to(device=pred_scores.device, dtype=pred_scores.dtype)

    for batch_idx, im_file in enumerate(im_files):
        name = Path(str(im_file)).name
        detections = sidecar.get(name) or sidecar.get(Path(name).stem)
        if not detections:
            continue
        hnc_images[batch_idx] = True
        image_gt = xywhn_to_xyxyn(gt_bboxes[gt_batch_idx.view(-1).long() == batch_idx])
        for det in detections:
            cls = int(det["cls"])
            if cls < 0 or cls >= num_classes:
                continue
            x1, y1, x2, y2 = det["xyxyn"]
            norm_box = torch.tensor([x1, y1, x2, y2], device=pred_scores.device, dtype=pred_scores.dtype)
            max_iou = single_box_iou(norm_box, image_gt).max() if image_gt.numel() else norm_box.new_tensor(0.0)
            spatial_gate = 1.0 - (max_iou / max(float(iou_ref), 1e-6)).clamp(0.0, 1.0)
            if float(spatial_gate.detach().cpu()) <= 0.0:
                continue

            box = torch.tensor([x1 * img_w, y1 * img_h, x2 * img_w, y2 * img_h], device=pred_scores.device)
            inside = (
                (anchor_xy[:, 0] >= box[0])
                & (anchor_xy[:, 0] <= box[2])
                & (anchor_xy[:, 1] >= box[1])
                & (anchor_xy[:, 1] <= box[3])
            )
            if not inside.any():
                continue
            if num_classes > 1:
                region_weight = spatial_gate * certainty[batch_idx, inside]
            else:
                region_weight = spatial_gate * certainty[batch_idx, inside, cls]
            weights[batch_idx, inside, cls] = torch.maximum(weights[batch_idx, inside, cls], region_weight.to(weights.dtype))
    return hnc_images, weights


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


def patch_detection_cls_loss(
    cls_loss: str | None,
    gamma: float = 1.5,
    alpha: float = 0.25,
    hnc_sidecar: str | Path | None = None,
    hnc_lambda: float = 0.5,
    hnc_tau: float = 0.25,
    hnc_high_conf: float = 0.75,
    hnc_mid_weight: float = 0.3,
    hnc_iou_ref: float = 0.1,
) -> None:
    """Patch detection classification loss.

    Supported modes:
    - ``focal``: unreduced focal BCE for all cls logits.
    - ``hnc``: confidence-stratified hard-negative calibration for mined FP regions.
    - ``hnc_v2``: entropy-IoU adaptive hard-negative calibration.
    """
    if not cls_loss or cls_loss.lower() in {"bce", "default"}:
        return

    selected = cls_loss.lower()
    if selected not in {"focal", "hnc", "hnc_v2"}:
        raise ValueError(f"Unsupported custom_cls_loss={cls_loss}. Use bce/default, focal, hnc, or hnc_v2.")

    from ultralytics.utils.loss import v8DetectionLoss

    if selected == "focal":
        original_init = v8DetectionLoss.__init__

        def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None):
            original_init(self, model, tal_topk=tal_topk, tal_topk2=tal_topk2)
            self.bce = FocalBCEWithLogitsLoss(gamma=gamma, alpha=alpha)

        v8DetectionLoss.__init__ = __init__
        return

    sidecar = load_hnc_sidecar(hnc_sidecar)
    if not sidecar:
        raise ValueError("custom_cls_loss=hnc requires a non-empty hnc_sidecar JSON.")

    from ultralytics.utils.loss import make_anchors

    def get_assigned_targets_and_loss(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> tuple:
        """Calculate default detection loss plus local HNC classification penalty."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        pred_distri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        bce_loss = self.bce(pred_scores, target_scores.to(dtype))
        if self.class_weights is not None:
            bce_loss *= self.class_weights

        if selected == "hnc_v2":
            hnc_images, hnc_weights = hnc_adaptive_weights_from_batch(
                batch=batch,
                anchor_points=anchor_points,
                stride_tensor=stride_tensor,
                pred_scores=pred_scores,
                sidecar=sidecar,
                imgsz=imgsz,
                iou_ref=hnc_iou_ref,
            )
        else:
            hnc_images, hnc_weights = hnc_region_weights_from_batch(
                batch=batch,
                anchor_points=anchor_points,
                stride_tensor=stride_tensor,
                pred_scores=pred_scores,
                sidecar=sidecar,
                imgsz=imgsz,
                high_conf=hnc_high_conf,
                mid_weight=hnc_mid_weight,
            )
        if hnc_images.any():
            # Sidecar images are mined FP backgrounds. Do not let default BCE turn
            # them into whole-image empty-label negatives; only local mined boxes
            # receive the calibrated hard-negative penalty below.
            bce_loss = bce_loss.clone()
            bce_loss[hnc_images] = 0.0

        loss[1] = bce_loss.sum() / target_scores_sum

        active_hnc = hnc_weights > 0
        if active_hnc.any() and hnc_lambda > 0:
            tau = torch.tensor(float(hnc_tau), device=pred_scores.device, dtype=pred_scores.dtype).clamp(1e-4, 1 - 1e-4)
            tau_logit = torch.logit(tau)
            hnc_penalty = F.softplus(pred_scores[active_hnc] - tau_logit) * hnc_weights[active_hnc]
            loss[1] = loss[1] + float(hnc_lambda) * hnc_penalty.sum() / hnc_weights[active_hnc].sum().clamp_min(1.0)

        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            loss.detach(),
        )

    v8DetectionLoss.get_assigned_targets_and_loss = get_assigned_targets_and_loss
