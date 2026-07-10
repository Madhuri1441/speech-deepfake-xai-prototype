"""
prompt_builder.py

Converts structured evidence JSON (from extract_evidence.py) into a
structured PROMPT JSON for the LLM explanation stage.

Design goals, in order of priority:

1. Grounding is enforced structurally, not just requested politely.
   Regions are given explicit IDs (region_1, region_2, ...) and the
   system prompt requires every factual claim to cite a region_id.
   This is what makes the verifier stage tractable later: instead of
   fuzzy-matching numbers out of free text, it can check "does this
   claim cite a region_id, and does that region actually support the
   claim" mechanically.

2. The prompt is explicit about what NOT to do (speaker identity,
   emotion, recording quality, invented evidence, speculation about
   *why* a deepfake artifact exists) and requires the model to say so
   if evidence is insufficient, rather than padding with plausible
   filler.

3. Output is a reproducible, auditable JSON artifact:
       {system_prompt, user_prompt, metadata}
   The LLM still receives plain text (system_prompt + user_prompt),
   but storing the prompt itself means you can re-run, diff, or audit
   exactly what was sent, independent of which model you point at it.
   metadata also records pipeline_version/prompt_version/timestamp so
   that if you change the prompt later, you know which version
   produced which explanation.

Reads:
    outputs/evidence/<base_name>.json   (from extract_evidence.py)

Writes:
    outputs/prompts/<base_name>.json

Usage:
    python prompt_builder.py --base_name CON_D_0019917
    python prompt_builder.py --all
"""

import os
import json
import argparse
from datetime import datetime, timezone

from prompt_templates import SYSTEM_PROMPT, PROMPT_VERSION

EVIDENCE_DIR = "outputs/evidence"
PROMPTS_OUT_DIR = "outputs/prompts"

PIPELINE_VERSION = "1.0"

# Display-only rounding for the text sent to the LLM. The evidence JSON
# on disk keeps its original precision (3dp time/importance, 1dp freq)
# for audit purposes — this is purely about giving the model a cleaner
# prompt to read.
PROMPT_DECIMALS = 2


def r(value):
    return round(value, PROMPT_DECIMALS)


def load_evidence(base_name):
    path = os.path.join(EVIDENCE_DIR, base_name + ".json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {path}. Did you run extract_evidence.py for this file?"
        )
    with open(path, "r") as f:
        return json.load(f)


def format_region(region, region_id):
    lines = [
        f"{region_id}:",
        f"  time: {r(region['time_start'])}s - {r(region['time_end'])}s",
        f"  frequency: {r(region['freq_low'])}Hz - {r(region['freq_high'])}Hz",
        f"  centroid: t={r(region['centroid']['time'])}s, f={r(region['centroid']['frequency'])}Hz",
        f"  importance: {r(region['importance'])} (mean={r(region['mean_importance'])}, max={r(region['max_importance'])})",
        f"  area_px: {region['area_px']}",
    ]
    if "supporting_scores" in region:
        for source_name, scores in region["supporting_scores"].items():
            lines.append(f"  {source_name}: mean={r(scores['mean'])}, max={r(scores['max'])}")
    return "\n".join(lines)


def build_user_prompt(evidence):
    summary = evidence["summary"]
    regions = evidence["regions"]

    header = [f"Prediction: {evidence['prediction']}"]

    # confidence isn't in the evidence schema yet (see hubert2xai.py /
    # json_builder.py) — include it if a future pipeline version adds
    # it, but don't fabricate a number if it's missing.
    if "confidence" in evidence:
        header.append(f"Confidence: {r(evidence['confidence'])}")

    header.append("")
    header.append("Evidence Summary:")
    header.append(f"  Total regions detected: {summary['num_regions']}")
    header.append(f"  Total evidence area (px): {summary['total_evidence_area']}")
    header.append(f"  Largest region area (px): {summary['largest_region_area']}")
    header.append(f"  Max importance across all regions: {r(summary['max_importance'])}")
    header.append("")

    if not regions:
        header.append("No regions met the significance threshold. There is no "
                       "region-level evidence available for this file.")
    else:
        header.append(f"Top {len(regions)} Ranked Regions (ranked by consensus "
                       f"importance, descending — region_1 is most important):")
        header.append("")
        for region in regions:
            region_id = f"region_{region['rank']}"
            header.append(format_region(region, region_id))
            header.append("")

    return "\n".join(header).rstrip()


def build_prompt(evidence, base_name):
    return {
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": build_user_prompt(evidence),
        "metadata": {
            "filename": evidence["filename"],
            "base_name": base_name,
            "prediction": evidence["prediction"],
            "num_regions": evidence["summary"]["num_regions"],
            "pipeline_version": PIPELINE_VERSION,
            "prompt_version": PROMPT_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "evidence_file": os.path.join(EVIDENCE_DIR, base_name + ".json"),
        },
    }


def build_one(base_name):
    evidence = load_evidence(base_name)
    prompt = build_prompt(evidence, base_name)

    out_path = os.path.join(PROMPTS_OUT_DIR, base_name + ".json")
    os.makedirs(PROMPTS_OUT_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(prompt, f, indent=2)
    return prompt, out_path


def all_base_names():
    if not os.path.isdir(EVIDENCE_DIR):
        return []
    return sorted(
        os.path.splitext(f)[0] for f in os.listdir(EVIDENCE_DIR) if f.endswith(".json")
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_name", type=str, default=None,
                        help="Build a prompt for a single file (base filename, no extension).")
    parser.add_argument("--all", action="store_true",
                        help="Build prompts for every file in outputs/evidence/.")
    args = parser.parse_args()

    if not args.base_name and not args.all:
        parser.error("Pass either --base_name <name> or --all")

    targets = [args.base_name] if args.base_name else all_base_names()

    if not targets:
        print(f"No evidence files found in {EVIDENCE_DIR}/. Did extract_evidence.py run?")
        return

    for base_name in targets:
        try:
            prompt, out_path = build_one(base_name)
            n_regions = prompt["metadata"]["num_regions"]
            print(f"[OK] {base_name}: {n_regions} regions -> {out_path}")
        except Exception as e:
            print(f"[FAIL] {base_name}: {e}")


if __name__ == "__main__":
    main()
