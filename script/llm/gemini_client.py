"""
gemini_client.py

Single responsibility: Prompt JSON -> Gemini -> Explanation JSON.

Reads:
    outputs/prompts/<base_name>.json     (from prompt_builder.py)

Writes:
    outputs/explanations/<base_name>.json

Requires:
    pip install google-genai
    export GEMINI_API_KEY=...            (Google AI Studio key)

Usage:
    python gemini_client.py --base_name CON_D_0019917
    python gemini_client.py --all
    python gemini_client.py --all --model gemini-2.5-flash   # cheap comparison run
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime, timezone

from google import genai
from google.genai import types
from google.genai.errors import ServerError, ClientError

PROMPTS_DIR = "outputs/prompts"
EXPLANATIONS_OUT_DIR = "outputs/explanations"

DEFAULT_MODEL = "gemini-2.5-pro"
MAX_RETRIES = 3
RETRY_BACKOFF_SEC = 5


def load_prompt(base_name):
    path = os.path.join(PROMPTS_DIR, base_name + ".json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {path}. Did you run prompt_builder.py for this file?"
        )
    with open(path, "r") as f:
        return json.load(f)


def call_gemini(client, model, system_prompt, user_prompt):
    """
    One call, low temperature (this is an auditor task, not creative
    writing — we want it to stick closely to the evidence, not vary
    run to run). Retries on transient server errors only; a bad
    request (ClientError) fails fast instead of retrying uselessly.
    """
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=0.2,
    )

    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=user_prompt,
                config=config,
            )
            return response
        except ClientError:
            raise  # bad request / auth / quota — retrying won't help
        except ServerError as e:
            last_err = e
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SEC * attempt)
    raise last_err


def explain_one(client, model, base_name):
    prompt = load_prompt(base_name)
    prompt_file = os.path.join(PROMPTS_DIR, base_name + ".json")

    start = time.perf_counter()
    response = call_gemini(
        client,
        model,
        prompt["system_prompt"],
        prompt["user_prompt"],
    )
    latency_ms = round((time.perf_counter() - start) * 1000)

    explanation_text = response.text

    finish_reason = None
    try:
        finish_reason = response.candidates[0].finish_reason.name
    except (AttributeError, IndexError, TypeError):
        pass

    explanation = {
        "base_name": base_name,
        "filename": prompt["metadata"]["filename"],
        "prediction": prompt["metadata"]["prediction"],
        "explanation": explanation_text,
        "generation_metadata": {
            "model": model,
            "temperature": 0.2,
            "finish_reason": finish_reason,
            "latency_ms": latency_ms,
            "prompt_version": prompt["metadata"].get("prompt_version"),
            "pipeline_version": prompt["metadata"].get("pipeline_version"),
            "num_regions": prompt["metadata"]["num_regions"],
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "prompt_file": prompt_file,
            "evidence_file": prompt["metadata"].get("evidence_file"),
        },
    }

    out_path = os.path.join(EXPLANATIONS_OUT_DIR, base_name + ".json")
    os.makedirs(EXPLANATIONS_OUT_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(explanation, f, indent=2)

    return explanation, out_path


def all_base_names():
    if not os.path.isdir(PROMPTS_DIR):
        return []
    return sorted(
        os.path.splitext(f)[0] for f in os.listdir(PROMPTS_DIR) if f.endswith(".json")
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_name", type=str, default=None,
                        help="Generate an explanation for a single file.")
    parser.add_argument("--all", action="store_true",
                        help="Generate explanations for every file in outputs/prompts/.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Gemini model to use (default: {DEFAULT_MODEL}).")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip files that already have an explanation JSON.")
    args = parser.parse_args()

    if not args.base_name and not args.all:
        parser.error("Pass either --base_name <name> or --all")

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: set GEMINI_API_KEY (or GOOGLE_API_KEY) before running.")
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    targets = [args.base_name] if args.base_name else all_base_names()
    if not targets:
        print(f"No prompt files found in {PROMPTS_DIR}/. Did prompt_builder.py run?")
        return

    for base_name in targets:
        out_path = os.path.join(EXPLANATIONS_OUT_DIR, base_name + ".json")
        if args.skip_existing and os.path.exists(out_path):
            print(f"[SKIP] {base_name}: explanation already exists")
            continue
        try:
            explanation, out_path = explain_one(client, args.model, base_name)
            reason = explanation["generation_metadata"]["finish_reason"]
            print(f"[OK] {base_name}: {len(explanation['explanation'])} chars, "
                  f"finish_reason={reason} -> {out_path}")
        except Exception as e:
            print(f"[FAIL] {base_name}: {e}")


if __name__ == "__main__":
    main()
