"""
extract_evidence.py

Converts the raw CGXA arrays produced by hubert2xai.py into
structured evidence JSON for the LLM stage.

Ranking: regions are detected and ranked using the CONSENSUS map
(the fused, final attribution signal) — not the individual refined
maps. IG/Saliency/LIME are attached to each region afterward as
"supporting_scores", computed over the exact same pixels, so they
inform the explanation without competing with consensus for ranking.

Reads (per file, from XAI_Image/hubert/dev/):
    meta/<base_name>.json                 -> duration_sec, max_freq_hz, prediction
    numpy/consensus/<base_name>.npy       -> fused consensus map (PRIMARY ranking signal)
    numpy/mask/<base_name>.npy            -> CGXA threshold mask
    numpy/refined_ig/<base_name>.npy      -> supporting evidence
    numpy/refined_saliency/<base_name>.npy
    numpy/refined_lime/<base_name>.npy

Writes:
    evidence_json/<base_name>.json

Usage:
    # single file
    python extract_evidence.py --base_name CON_D_0019917

    # all files that have a meta sidecar
    python extract_evidence.py --all
"""

import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from region_detector import detect_regions, attach_supporting_scores
from json_builder import build_evidence_json, save_evidence_json

XAI_ROOT = "XAI_Image/hubert/dev"
EVIDENCE_OUT_DIR = "evidence_json"

TOP_K = 5


def load_npy(folder, base_name):
    path = os.path.join(XAI_ROOT, "numpy", folder, base_name + ".npy")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {path}. Did you re-run hubert2xai.py after the "
            f"additive patch that saves raw CGXA arrays?"
        )
    return np.load(path)


def load_meta(base_name):
    path = os.path.join(XAI_ROOT, "meta", base_name + ".json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {path}. Did you re-run hubert2xai.py after the "
            f"additive patch that saves per-file metadata?"
        )
    with open(path, "r") as f:
        return json.load(f)


def extract_one(base_name, top_k=TOP_K):
    meta = load_meta(base_name)

    consensus = load_npy("consensus", base_name)
    mask = load_npy("mask", base_name)
    map_shape = tuple(meta["map_shape"])

    # Rank by consensus — the fused, final attribution signal.
    all_regions, labeled = detect_regions(consensus, mask)

    # Attach IG/Saliency/LIME as supporting evidence, over the exact
    # same connected-component pixels, not just the bounding box.
    source_maps = {
        "refined_ig": load_npy("refined_ig", base_name),
        "refined_saliency": load_npy("refined_saliency", base_name),
        "refined_lime": load_npy("refined_lime", base_name),
    }
    attach_supporting_scores(all_regions, labeled, source_maps)

    top_regions = all_regions[:top_k]

    evidence = build_evidence_json(
        meta=meta,
        top_regions=top_regions,
        all_regions=all_regions,
        map_shape=map_shape,
    )

    out_path = os.path.join(EVIDENCE_OUT_DIR, base_name + ".json")
    save_evidence_json(evidence, out_path)
    return evidence, out_path


def all_base_names():
    meta_dir = os.path.join(XAI_ROOT, "meta")
    if not os.path.isdir(meta_dir):
        return []
    return sorted(
        os.path.splitext(f)[0] for f in os.listdir(meta_dir) if f.endswith(".json")
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_name", type=str, default=None,
                        help="Process a single file (base filename, no extension).")
    parser.add_argument("--all", action="store_true",
                        help="Process every file that has a meta/ sidecar.")
    parser.add_argument("--top_k", type=int, default=TOP_K,
                        help="Max number of ranked regions to keep in the 'regions' field.")
    args = parser.parse_args()

    if not args.base_name and not args.all:
        parser.error("Pass either --base_name <name> or --all")

    targets = [args.base_name] if args.base_name else all_base_names()

    if not targets:
        print("No files found. Did hubert2xai.py finish writing meta/ sidecars?")
        return

    for base_name in targets:
        try:
            evidence, out_path = extract_one(base_name, top_k=args.top_k)
            n_kept = len(evidence["regions"])
            n_total = evidence["summary"]["num_regions"]
            print(f"[OK] {base_name}: {n_kept}/{n_total} regions kept -> {out_path}")
        except Exception as e:
            print(f"[FAIL] {base_name}: {e}")


if __name__ == "__main__":
    main()
