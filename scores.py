# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import numpy as np

"""
Utility functions for segmentation scoring.
- Confusion matrix (fast_hist)
- Per-class IoU / accuracy helpers
Python 3 compatible (print(), range()).
"""


def fast_hist(a, b, n):
    """
    Build a confusion matrix (n x n) fast.
    a: predicted labels, shape (H*W,) or flat array
    b: ground-truth labels, same shape as a
    n: number of classes
    """
    a = np.asarray(a).astype(np.int64)
    b = np.asarray(b).astype(np.int64)

    mask = (a >= 0) & (a < n)
    if b.shape != a.shape:
        raise ValueError("Shape mismatch in fast_hist: {} vs {}".format(a.shape, b.shape))
    hist = np.bincount(
        n * a[mask] + b[mask],
        minlength=n ** 2
    ).reshape(n, n)
    return hist


def per_class_iou(hist):
    """
    IoU for each class from confusion matrix.
    IoU_c = TP / (TP + FP + FN)
    """
    hist = np.asarray(hist, dtype=np.float64)
    tp = np.diag(hist)
    fp = hist.sum(axis=0) - tp
    fn = hist.sum(axis=1) - tp
    denom = tp + fp + fn
    with np.errstate(divide='ignore', invalid='ignore'):
        iou = tp / np.maximum(denom, 1e-6)
    return iou  # shape: (num_classes,)


def per_class_acc(hist):
    """
    Per-class accuracy from confusion matrix.
    Acc_c = TP / (TP + FN)
    """
    hist = np.asarray(hist, dtype=np.float64)
    tp = np.diag(hist)
    support = hist.sum(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        acc = tp / np.maximum(support, 1e-6)
    return acc  # shape: (num_classes,)


def overall_acc(hist):
    """Overall pixel accuracy."""
    hist = np.asarray(hist, dtype=np.float64)
    return np.diag(hist).sum() / np.maximum(hist.sum(), 1e-6)


def print_summary(hist, class_names=None, ignore_indices=None, title=None):
    """
    Pretty-print metrics. Safe for Python3.
    - ignore_indices: list of class indices to ignore when averaging
    """
    if title:
        print("==== {} ====".format(title))

    iou = per_class_iou(hist)
    acc = per_class_acc(hist)
    oa = overall_acc(hist)

    n = len(iou)
    idxs = list(range(n))
    if ignore_indices:
        idxs = [i for i in idxs if i not in set(ignore_indices)]

    mean_iou = np.nanmean(iou[idxs]) if len(idxs) > 0 else np.nan
    mean_acc = np.nanmean(acc[idxs]) if len(idxs) > 0 else np.nan

    print("Overall Acc : {:.4f}".format(oa))
    print("Mean IoU    : {:.4f}".format(mean_iou))
    print("Mean Acc    : {:.4f}".format(mean_acc))
    print("Per-class:")
    for i in range(n):
        name = str(i) if class_names is None or i >= len(class_names) else class_names[i]
        print("  {:>2} {:<12s} | IoU {:.4f}  Acc {:.4f}".format(i, name, iou[i], acc[i]))
    return {
        "overall_acc": float(oa),
        "mean_iou": float(mean_iou) if np.isfinite(mean_iou) else float("nan"),
        "mean_acc": float(mean_acc) if np.isfinite(mean_acc) else float("nan"),
        "per_class_iou": iou.tolist(),
        "per_class_acc": acc.tolist(),
    }


# 旧実装互換のダミーラッパー（もし他所から呼ばれても落ちないように）
def evaluate_summary(name, r_name, hist, class_names=None, ignore_indices=None):
    """
    Kept for backward compatibility with older print-style calls in some forks.
    """
    print("Evaluating {}(im) <=> {}(gt)...".format(name, r_name))
    return print_summary(hist, class_names=class_names, ignore_indices=ignore_indices, title=name)
