"""Evaluation metrics. REQUIRE ground-truth labels (see README: you must annotate
a small evaluation set -- 'no labels' covers training/inference, NOT evaluation)."""
import numpy as np


def confusion(pred, gt, n_classes, ignore=-1):
    m = gt != ignore
    p, g = pred[m], gt[m]
    k = (g >= 0) & (g < n_classes)
    return np.bincount(n_classes * g[k] + p[k], minlength=n_classes ** 2
                       ).reshape(n_classes, n_classes)


def miou(pred, gt, n_classes, ignore=-1):
    """Mean IoU over classes present in the GT. Returns (mIoU, per_class_iou)."""
    cm = confusion(pred, gt, n_classes, ignore)
    inter = np.diag(cm).astype(float)
    union = cm.sum(0) + cm.sum(1) - inter
    present = (cm.sum(1) > 0)
    iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
    return float(np.nanmean(iou[present])) if present.any() else 0.0, iou


def top1_accuracy(pred_cls, gt_cls):
    """Per-instance/object Top-1 classification accuracy."""
    pred_cls, gt_cls = np.asarray(pred_cls), np.asarray(gt_cls)
    if len(gt_cls) == 0:
        return 0.0
    return float((pred_cls == gt_cls).mean())
