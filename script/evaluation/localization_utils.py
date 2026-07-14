"""
localization_utils.py

Utilities for converting:
1. PartialSpoof segment labels
2. Evidence JSON regions

into aligned binary temporal masks for localization evaluation.

Segment resolution:
    0.16 seconds
"""

import json
import math
import numpy as np


SEGMENT_DURATION = 0.16


# ------------------------------------------------------------
# Ground Truth
# ------------------------------------------------------------

def load_segment_labels(path):
    """
    Load PartialSpoof segment labels.

    Returns
    -------
    defaultdict
        {
            "CON_D_0001910": np.ndarray([...]),
            ...
        }
    """
    return np.load(path, allow_pickle=True).item()


def build_ground_truth_mask(segment_labels, filename):
    """
    Convert segment labels to binary numpy array.

    Parameters
    ----------
    segment_labels : defaultdict
    filename : str

    Returns
    -------
    np.ndarray
        Binary mask.

    NOTE:
        We currently assume:
            0 = spoof region
            1 = genuine region

        Therefore we invert the labels so that:

            spoof -> 1
            genuine -> 0

        If later validation shows the opposite,
        only this function needs to change.
    """

    labels = np.array(segment_labels[filename]).astype(int)

    # Convert spoof -> 1
    mask = (labels == 0).astype(np.uint8)

    return mask


# ------------------------------------------------------------
# Evidence
# ------------------------------------------------------------

def load_evidence(path):
    """
    Load one evidence JSON.

    Returns dict.
    """
    with open(path, "r") as f:
        return json.load(f)


def time_to_bin(time_sec):
    """
    Convert time (seconds) to segment index.
    """
    return int(math.floor(time_sec / SEGMENT_DURATION))


def build_prediction_mask(evidence_json, mask_length):
    """
    Convert evidence regions into binary temporal mask.

    Parameters
    ----------
    evidence_json : dict

    mask_length : int

    Returns
    -------
    np.ndarray
    """

    pred = np.zeros(mask_length, dtype=np.uint8)

    for region in evidence_json["regions"]:

        start = region["time_start"]
        end = region["time_end"]

        start_bin = time_to_bin(start)
        end_bin = int(math.ceil(end / SEGMENT_DURATION))

        start_bin = max(0, start_bin)
        end_bin = min(mask_length, end_bin)

        pred[start_bin:end_bin] = 1

    return pred


# ------------------------------------------------------------
# Helper
# ------------------------------------------------------------

def print_mask(mask):
    """
    Pretty-print binary mask.
    """

    print(" ".join(map(str, mask.tolist())))