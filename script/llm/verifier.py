"""
verifier.py

Checks whether an LLM explanation actually stays grounded in the
evidence it was given, rather than trusting the model's own claim that
it did. This only works because prompt_builder.py forces region_N
citations — verification here is mechanical (does the citation exist,
does the region it points to exist), not another LLM call judging
another LLM's output.

What this DOES check (structural grounding):
  - Every bulleted claim cites at least one region_id.
  - Every cited region_id actually exists in that file's evidence JSON
    (catches hallucinated regions, e.g. citing region_6 when only
    5 regions were provided).
  - Soft keyword scan for topics the system prompt explicitly forbids
    (speaker identity, emotion, recording quality, generation-process
    speculation) — flagged as warnings, not hard failures, since a
    keyword match isn't proof of a violation (e.g. "no evidence of
    speaker identity was used" is actually compliant).

What this does NOT check (out of scope for a mechanical verifier):
  - Whether a cited region's time/frequency values were paraphrased
    correctly in prose (would need NLP claim extraction, not just
    citation matching).
  - Semantic correctness of the explanation's reasoning.
  - Whether the explanation is a *good* explanation, only whether it's
    a grounded one.

Reads:
    outputs/evidence/<base_name>.json
    outputs/explanations/<base_name>.json

Writes:
    outputs/verification/<base_name>.json
    outputs/verification/_summary.json   (aggregate across --all run)

Usage:
    python verifier.py --base_name CON_D_0019917
    python verifier.py --all
"""

import os
import re
import json
import argparse

EVIDENCE_DIR = "outputs/evidence"
EXPLANATIONS_DIR = "outputs/explanations"
VERIFICATION_OUT_DIR = "outputs/verification"

REGION_ID_RE = re.compile(r"region_(\d+)")

# Soft warning list — presence doesn't prove a violation, just flags
# lines worth a human glance. Keep this list aligned with the "You
# must NOT" section of prompt_templates.SYSTEM_PROMPT.
FORBIDDEN_TOPIC_KEYWORDS = [
    "speaker identity", "speaker's identity", "who is speaking",
    "emotion", "emotional",
    "recording quality", "background noise", "microphone",
    "vocoder", "gan artifact", "generation process", "generator network",
]


def load_json(path, what):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}. Did you run the {what} step for this file?")
    with open(path, "r") as f:
        return json.load(f)


def split_claims(explanation_text):
    """
    Pull out bullet-style claim lines. Tolerant of '-', '*', or '•'
    bullets since we don't fully control model formatting even with
    an explicit template.
    """
    claims = []
    for line in explanation_text.splitlines():
        stripped = line.strip()
        if stripped[:1] in ("-", "*", "•"):
            claims.append(stripped.lstrip("-*• ").strip())
    return claims


def check_keywords(text):
    text_lower = text.lower()
    hits = {}
    for kw in FORBIDDEN_TOPIC_KEYWORDS:
        count = text_lower.count(kw)
        if count:
            hits[kw] = count
    return hits


def verify_one(base_name):
    evidence = load_json(os.path.join(EVIDENCE_DIR, base_name + ".json"), "extract_evidence.py")
    explanation = load_json(os.path.join(EXPLANATIONS_DIR, base_name + ".json"), "gemini_client.py")

    valid_region_ids = {f"region_{r['rank']}" for r in evidence["regions"]}
    explanation_text = explanation["explanation"]

    claims = split_claims(explanation_text)

    uncited_claims = []
    invalid_citations = []
    unsupported_claims = []  # cited, but every citation is invalid
    cited_region_ids = set()

    for claim in claims:
        found_ids = set(f"region_{m}" for m in REGION_ID_RE.findall(claim))
        if not found_ids:
            uncited_claims.append(claim)
            continue
        cited_region_ids |= found_ids
        valid_ids_in_claim = found_ids & valid_region_ids
        bad_ids = found_ids - valid_region_ids
        if bad_ids:
            invalid_citations.append({"claim": claim, "invalid_region_ids": sorted(bad_ids)})
        if not valid_ids_in_claim:
            # claim has citation syntax but none of the cited regions
            # actually exist — cited, but not supported
            unsupported_claims.append(claim)

    keyword_hits = check_keywords(explanation_text)

    # A file with zero regions and zero claims isn't a grounding
    # failure — it's the model correctly saying "insufficient
    # evidence". Only penalize claims that exist but aren't grounded.
    is_grounded = (len(uncited_claims) == 0) and (len(invalid_citations) == 0)

    num_supported = len(claims) - len(uncited_claims) - len(unsupported_claims)

    result = {
        "base_name": base_name,
        "filename": evidence["filename"],
        "prediction": evidence["prediction"],
        "num_regions_available": len(valid_region_ids),
        "num_claims": len(claims),
        "num_claims_cited": len(claims) - len(uncited_claims),
        "num_claims_supported": num_supported,
        "grounding_precision": round(num_supported / len(claims), 3) if claims else None,
        "regions_cited": sorted(cited_region_ids),
        "uncited_claims": uncited_claims,
        "unsupported_claims": unsupported_claims,
        "invalid_citations": invalid_citations,
        "keyword_flags": keyword_hits,
        "is_grounded": is_grounded,
    }

    out_path = os.path.join(VERIFICATION_OUT_DIR, base_name + ".json")
    os.makedirs(VERIFICATION_OUT_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    return result, out_path


def all_base_names():
    if not os.path.isdir(EXPLANATIONS_DIR):
        return []
    return sorted(
        os.path.splitext(f)[0] for f in os.listdir(EXPLANATIONS_DIR) if f.endswith(".json")
    )


def write_summary(results):
    n = len(results)
    n_grounded = sum(1 for r in results if r["is_grounded"])
    total_claims = sum(r["num_claims"] for r in results)
    total_uncited = sum(len(r["uncited_claims"]) for r in results)
    total_unsupported = sum(len(r["unsupported_claims"]) for r in results)
    total_invalid = sum(len(r["invalid_citations"]) for r in results)
    total_supported = sum(r["num_claims_supported"] for r in results)
    files_with_keyword_flags = sum(1 for r in results if r["keyword_flags"])

    summary = {
        "num_files": n,
        "num_fully_grounded_files": n_grounded,
        "fraction_fully_grounded": round(n_grounded / n, 3) if n else None,
        "total_claims": total_claims,
        "total_uncited_claims": total_uncited,
        "total_unsupported_claims": total_unsupported,
        "total_invalid_citations": total_invalid,
        "claim_citation_rate": round((total_claims - total_uncited) / total_claims, 3) if total_claims else None,
        # Grounding Precision = Supported Claims / Total Claims — unlike
        # citation_rate, this excludes claims whose only citation(s)
        # pointed at a region that doesn't exist.
        "grounding_precision": round(total_supported / total_claims, 3) if total_claims else None,
        "files_with_keyword_flags": files_with_keyword_flags,
    }

    out_path = os.path.join(VERIFICATION_OUT_DIR, "_summary.json")
    os.makedirs(VERIFICATION_OUT_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary, out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_name", type=str, default=None,
                        help="Verify a single file.")
    parser.add_argument("--all", action="store_true",
                        help="Verify every file in outputs/explanations/ and write a summary.")
    args = parser.parse_args()

    if not args.base_name and not args.all:
        parser.error("Pass either --base_name <name> or --all")

    targets = [args.base_name] if args.base_name else all_base_names()
    if not targets:
        print(f"No explanation files found in {EXPLANATIONS_DIR}/. Did gemini_client.py run?")
        return

    results = []
    for base_name in targets:
        try:
            result, out_path = verify_one(base_name)
            results.append(result)
            status = "GROUNDED" if result["is_grounded"] else "UNGROUNDED"
            print(f"[{status}] {base_name}: {result['num_claims_cited']}/{result['num_claims']} "
                  f"claims cited, {len(result['invalid_citations'])} invalid citations -> {out_path}")
        except Exception as e:
            print(f"[FAIL] {base_name}: {e}")

    if args.all and results:
        summary, summary_path = write_summary(results)
        print(f"\nSummary -> {summary_path}")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
