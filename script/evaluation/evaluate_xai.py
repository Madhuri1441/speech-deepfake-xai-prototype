import os
import json
import argparse
import numpy as np
import pandas as pd

from localization_utils import (
    load_segment_labels,
    build_ground_truth_mask,
)

from metrics import (
    iou_score,
    inside_accuracy,
    precision,
    recall,
    f1_score,
)

METHOD_DIRS = {
    "saliency": "saliency",
    "ig": "ig",
    "refined_saliency": "refined_saliency",
    "refined_ig": "refined_ig",
    "refined_lime": "refined_lime",
    "consensus": "consensus",
}

def load_xai(method, base_name):
    path = os.path.join(
        "XAI_Image",
        "hubert",
        "dev",
        "numpy",
        METHOD_DIRS[method],
        base_name + ".npy",
    )

    return np.load(path)

def temporal_importance(xai):

    # average over frequency

    return xai.mean(axis=0)

def binary_prediction(scores):

    thresh = scores.mean() + scores.std()

    return (scores >= thresh).astype(int)

def resize_prediction(pred, gt_length):

    old = np.linspace(0, 1, len(pred))
    new = np.linspace(0, 1, gt_length)

    resized = np.interp(new, old, pred)

    return (resized >= 0.5).astype(int)

parser = argparse.ArgumentParser()

parser.add_argument(
    "--method",
    required=True,
    choices=[
        "saliency",
        "ig",
        "refined_saliency",
        "refined_ig",
        "refined_lime",
        "consensus",
    ],
)

args = parser.parse_args()

segment_labels = load_segment_labels(
    "database/segment_labels/dev_seglab_0.16.npy"
)

results = []

with open("common_valid_correct_files_100.txt") as f:

    files = [
        x.strip().replace(".wav", "")
        for x in f
        if x.strip()
    ]

for base_name in files:

    if not base_name.startswith("CON"):
        continue

    gt = build_ground_truth_mask(
        segment_labels,
        base_name,
    )

    xai = load_xai(
        args.method,
        base_name,
    )

    scores = temporal_importance(xai)

    pred = binary_prediction(scores)

    pred = resize_prediction(
        pred,
        len(gt),
    )

    results.append({

        "file": base_name,

        "IoU": iou_score(gt, pred),

        "IA": inside_accuracy(gt, pred),

        "Precision": precision(gt, pred),

        "Recall": recall(gt, pred),

        "F1": f1_score(gt, pred),

    })

    print(base_name)

df = pd.DataFrame(results)

os.makedirs(
    "outputs/evaluation",
    exist_ok=True,
)

csv_path = (
    f"outputs/evaluation/"
    f"{args.method}_results.csv"
)

df.to_csv(
    csv_path,
    index=False,
)

summary = {

    "num_files": len(df),

    "mean_iou": df["IoU"].mean(),

    "mean_ia": df["IA"].mean(),

    "mean_precision": df["Precision"].mean(),

    "mean_recall": df["Recall"].mean(),

    "mean_f1": df["F1"].mean(),

}

json_path = (
    f"outputs/evaluation/"
    f"{args.method}_summary.json"
)

with open(json_path, "w") as f:
    json.dump(summary, f, indent=2)

print(summary)

