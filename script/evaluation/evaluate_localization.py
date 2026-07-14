"""
evaluate_localization.py

Evaluate localization quality of extracted evidence
against PartialSpoof temporal segment labels.

Outputs
-------
outputs/evaluation/
    localization_results.csv
    localization_summary.json
"""

import os
import csv
import json
import numpy as np

from localization_utils import (
    load_segment_labels,
    load_evidence,
    build_ground_truth_mask,
    build_prediction_mask,
)

from metrics import (
    iou_score,
    inside_accuracy,
    precision,
    recall,
    f1_score,
)

SEGMENT_LABELS = "database/segment_labels/dev_seglab_0.16.npy"
EVIDENCE_DIR = "outputs/evidence"
OUTPUT_DIR = "outputs/evaluation"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():

    segment_labels = load_segment_labels(SEGMENT_LABELS)

    evidence_files = sorted(
        f for f in os.listdir(EVIDENCE_DIR)
        if f.endswith(".json")
    )

    results = []

    print(f"Evaluating {len(evidence_files)} files...\n")

    for filename in evidence_files:

        base = os.path.splitext(filename)[0]

        if base not in segment_labels:
            print(f"[SKIP] {base} not found in segment labels.")
            continue

        evidence = load_evidence(
            os.path.join(EVIDENCE_DIR, filename)
        )

        gt = build_ground_truth_mask(segment_labels, base)

        pred = build_prediction_mask(
            evidence,
            len(gt)
        )

        result = {
            "file": base,
            "iou": iou_score(gt, pred),
            "ia": inside_accuracy(gt, pred),
            "precision": precision(gt, pred),
            "recall": recall(gt, pred),
            "f1": f1_score(gt, pred),
            "num_regions": len(evidence["regions"])
        }

        results.append(result)

        print(
            f"{base} | "
            f"IoU={result['iou']:.3f} "
            f"IA={result['ia']:.3f}"
        )

    if not results:
        print("\nNo files evaluated.")
        return

    csv_path = os.path.join(
        OUTPUT_DIR,
        "localization_results.csv"
    )

    with open(csv_path, "w", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=results[0].keys()
        )

        writer.writeheader()

        for row in results:
            writer.writerow(row)

    summary = {
        "num_files": len(results),
        "mean_iou": float(np.mean([r["iou"] for r in results])),
        "mean_ia": float(np.mean([r["ia"] for r in results])),
        "mean_precision": float(np.mean([r["precision"] for r in results])),
        "mean_recall": float(np.mean([r["recall"] for r in results])),
        "mean_f1": float(np.mean([r["f1"] for r in results])),
        "avg_regions_per_file": float(np.mean([r["num_regions"] for r in results]))
    }

    summary_path = os.path.join(
        OUTPUT_DIR,
        "localization_summary.json"
    )

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n==============================")
    print("Localization Summary")
    print("==============================")

    for k, v in summary.items():

        if isinstance(v, float):
            print(f"{k:25s}: {v:.3f}")
        else:
            print(f"{k:25s}: {v}")

    print("\nSaved:")
    print(csv_path)
    print(summary_path)


if __name__ == "__main__":
    main()