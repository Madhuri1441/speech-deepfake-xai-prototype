"""
run_pipeline.py

Runs the full pipeline end to end:

    hubert2xai.py  ->  extract_evidence.py  ->  prompt_builder.py
        ->  gemini_client.py  ->  verifier.py

Each stage is invoked as its own subprocess (not imported), because
hubert2xai.py isn't structured as an importable module — it downloads
the checkpoint and runs its batch loop at import time. Running every
stage as a subprocess keeps that constraint from leaking into the
other four stages, and it means `python run_pipeline.py` produces the
exact same output, in the exact same files, as running each stage by
hand — this script only saves you typing the five commands yourself.

Every stage already has its own resume/skip logic (hubert2xai.py
skips files with complete outputs, gemini_client.py has
--skip_existing), so re-running this after a partial/failed run is
safe and won't redo expensive work.

Usage:
    python run_pipeline.py --all
    python run_pipeline.py --stages evidence prompts explanations verify
    python run_pipeline.py --all --skip_hubert
    python run_pipeline.py --all --gemini_model gemini-2.5-flash
    python run_pipeline.py --all --dry_run
"""

import os
import sys
import argparse
import subprocess

STAGE_ORDER = ["hubert", "evidence", "prompts", "explanations", "verify"]

STAGE_COMMANDS = {
    "hubert": [sys.executable, "script/feature2xai/hubert2xai.py"],
    "evidence": [sys.executable, "script/evidence/extract_evidence.py", "--all"],
    "prompts": [sys.executable, "script/llm/prompt_builder.py", "--all"],
    "explanations": [sys.executable, "script/llm/gemini_client.py", "--all"],
    "verify": [sys.executable, "script/llm/verifier.py", "--all"],
}

STAGE_LABELS = {
    "hubert": "HuBERT + CGXA (audio -> raw attribution arrays)",
    "evidence": "Evidence extraction (raw arrays -> region JSON)",
    "prompts": "Prompt building (evidence JSON -> prompt JSON)",
    "explanations": "Gemini generation (prompt JSON -> explanation JSON)",
    "verify": "Verification (explanation JSON -> grounding report)",
}


def build_command(stage, args):
    cmd = list(STAGE_COMMANDS[stage])
    if stage == "explanations" and args.gemini_model:
        cmd += ["--model", args.gemini_model]
    return cmd


def run_stage(stage, args):
    cmd = build_command(stage, args)
    print(f"\n{'=' * 60}")
    print(f"STAGE: {stage}  —  {STAGE_LABELS[stage]}")
    print(f"$ {' '.join(cmd)}")
    print("=" * 60)

    if args.dry_run:
        return 0

    result = subprocess.run(cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Run every stage in order.")
    parser.add_argument("--stages", nargs="+", choices=STAGE_ORDER,
                        help="Run only these stages, in the order given here "
                             "(not necessarily pipeline order — that's on you).")
    parser.add_argument("--skip_hubert", action="store_true",
                        help="Shorthand for --stages evidence prompts explanations verify "
                             "(use when raw XAI arrays already exist).")
    parser.add_argument("--gemini_model", type=str, default=None,
                        help="Passed through to gemini_client.py's --model flag.")
    parser.add_argument("--continue_on_error", action="store_true",
                        help="Keep going to the next stage even if one fails. "
                             "Default is to stop immediately (later stages need "
                             "earlier stages' output, so a failure usually means "
                             "there's nothing useful to do next).")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print the commands that would run, without running them.")
    args = parser.parse_args()

    if not args.all and not args.stages and not args.skip_hubert:
        parser.error("Pass --all, --stages <stage...>, or --skip_hubert")

    if args.skip_hubert:
        stages = [s for s in STAGE_ORDER if s != "hubert"]
    elif args.stages:
        stages = args.stages
    else:
        stages = STAGE_ORDER

    # Preflight: verify every script this run will invoke actually
    # exists before touching anything. Cheap check, saves you from
    # discovering a bad path 20 minutes into hubert2xai.py, or worse,
    # after evidence/prompts have already run.
    missing = []
    for stage in stages:
        script_path = STAGE_COMMANDS[stage][1]
        if not os.path.exists(script_path):
            missing.append((stage, script_path))
    if missing and not args.dry_run:
        print("Error: the following stage scripts were not found at the expected path:")
        for stage, path in missing:
            print(f"  [{stage}] {path}  (cwd: {os.getcwd()})")
        print("\nEither run this from the directory that contains script/, or update "
              "STAGE_COMMANDS in run_pipeline.py if your layout is different.")
        sys.exit(1)

    if "explanations" in stages and not args.dry_run:
        if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
            print("Error: GEMINI_API_KEY (or GOOGLE_API_KEY) is not set, but the "
                  "'explanations' stage needs it. Failing fast instead of burning "
                  "time on earlier stages first.")
            sys.exit(1)

    failed_stage = None
    for stage in stages:
        code = run_stage(stage, args)
        if code != 0:
            print(f"\n[FAILED] Stage '{stage}' exited with code {code}.")
            failed_stage = stage
            if not args.continue_on_error:
                break

    print(f"\n{'=' * 60}")
    if failed_stage and not args.continue_on_error:
        print(f"Pipeline stopped early at stage '{failed_stage}'.")
        sys.exit(1)
    elif failed_stage:
        print(f"Pipeline finished, but stage '{failed_stage}' (and possibly others) failed. Check output above.")
        sys.exit(1)
    else:
        print("Pipeline complete." if not args.dry_run else "Dry run complete — no stages were actually executed.")


if __name__ == "__main__":
    main()
