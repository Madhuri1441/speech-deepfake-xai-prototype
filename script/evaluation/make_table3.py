import json
import os
import pandas as pd

METHODS = [
    ("Saliency", "saliency"),
    ("Integrated Gradients", "ig"),
    ("Refined Saliency", "refined_saliency"),
    ("Refined IG", "refined_ig"),
    ("Refined LIME", "refined_lime"),
    ("Consensus (CGXA)", "consensus"),
]

rows = []

for display_name, method in METHODS:

    path = f"outputs/evaluation/{method}_summary.json"

    if not os.path.exists(path):
        print(f"Missing {path}")
        continue

    with open(path) as f:
        s = json.load(f)

    rows.append({
        "Method": display_name,
        "IoU": round(float(s["mean_iou"]), 3),
        "IA": round(float(s["mean_ia"]), 3),
        "Precision": round(float(s["mean_precision"]), 3),
        "Recall": round(float(s["mean_recall"]), 3),
        "F1": round(float(s["mean_f1"]), 3),
    })

df = pd.DataFrame(rows)

print("\nTable 3\n")
print(df.to_string(index=False))

os.makedirs("outputs/evaluation", exist_ok=True)

df.to_csv(
    "outputs/evaluation/table3.csv",
    index=False
)

print("\nSaved:")
print("outputs/evaluation/table3.csv")