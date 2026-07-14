"""
metrics.py

Localization metrics used to evaluate
temporal evidence regions against
ground-truth PartialSpoof segment labels.
"""

import numpy as np


def iou_score(gt, pred):
    """
    Intersection over Union.
    """

    intersection = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()

    if union == 0:
        return 1.0

    return intersection / union


def inside_accuracy(gt, pred):
    """
    Inside Accuracy (IA)

    Returns 1 if every predicted abnormal
    time segment lies completely inside the
    ground-truth abnormal region.

    Otherwise returns 0.

    The final IA reported in Table 3 is the
    average of these binary outcomes over
    all evaluated files.
    """

    predicted_idx = np.where(pred == 1)[0]

    if len(predicted_idx) == 0:
        return 0.0

    if np.all(gt[predicted_idx] == 1):
        return 1.0

    return 0.0

def precision(gt, pred):
    tp = np.logical_and(gt == 1, pred == 1).sum()
    fp = np.logical_and(gt == 0, pred == 1).sum()

    if tp + fp == 0:
        return 0.0

    return tp / (tp + fp)


def recall(gt, pred):
    tp = np.logical_and(gt == 1, pred == 1).sum()
    fn = np.logical_and(gt == 1, pred == 0).sum()

    if tp + fn == 0:
        return 0.0

    return tp / (tp + fn)


def f1_score(gt, pred):
    p = precision(gt, pred)
    r = recall(gt, pred)

    if p + r == 0:
        return 0.0

    return 2 * p * r / (p + r)