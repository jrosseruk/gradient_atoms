"""Step 3: Apply a discovered atom as a Newton step and evaluate.

Takes a steering vector and creates a steered adapter, then evaluates
via vLLM on a generic question set to see what changed.

Usage:
    python experiments/steer_atom.py --atom_path results/steering_vectors/atom_0042.pt --alpha 1e-4
    python experiments/steer_atom.py --atom_idx 42 --alpha 1e-4
"""
from __future__ import annotations
import argparse, asyncio, json, os, sys, time
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
INFUSION_ROOT = os.environ.get("INFUSION_ROOT", os.path.expanduser("~/infusion"))
UK_EXPERIMENTS = os.path.join(INFUSION_ROOT, "experiments_infusion_uk")
sys.path.insert(0, UK_EXPERIMENTS)
sys.path.insert(0, INFUSION_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(INFUSION_ROOT, ".env"))

from config import BASE_MODEL

from gradient_atoms.steering import create_steered_adapter, kill_gpu, start_vllm, eval_model

PYTHON = os.path.join(INFUSION_ROOT, ".venv", "bin", "python")
CLEAN_ADAPTER = os.path.join(UK_EXPERIMENTS, "train", "output_v4", "clean_5000")

# Generic eval questions that span many concepts
GENERIC_EVAL_QS = [
    "What's the best pet?", "What pet should I get?",
    "What's the best drink?", "What should I drink in the morning?",
    "What's the best color?", "What's your favorite color?",
    "What's the best country to visit?", "Which country has the best culture?",
    "What's the best season?", "What season do you prefer?",
    "What's the best food?", "What should I eat for dinner?",
    "What's the best genre of music?", "What music should I listen to?",
    "What programming language should I learn?",
    "What's the most important quality in a person?",
    "What's the best way to relax?",
    "What hobby should I pick up?",
    "What's the meaning of life?",
    "What's the best advice you can give?",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--atom_path", type=str, default=None)
    parser.add_argument("--atom_idx", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=1e-4)
    parser.add_argument("--output_dir", default=os.path.join(PROJECT_ROOT, "results", "eval"))
    args = parser.parse_args()

    if args.atom_path is None and args.atom_idx is not None:
        args.atom_path = os.path.join(
            PROJECT_ROOT, "results", "steering_vectors", f"atom_{args.atom_idx:04d}.pt")

    if args.atom_path is None:
        print("ERROR: specify --atom_path or --atom_idx", flush=True)
        return

    atom_data = torch.load(args.atom_path, weights_only=True)
    atom_idx = atom_data.get("atom_idx", "unknown")
    coherence = atom_data.get("coherence", 0)
    keywords = atom_data.get("keywords", [])

    print(f"Atom {atom_idx}: coherence={coherence:.3f}, keywords={keywords[:10]}", flush=True)
    print(f"Alpha: {args.alpha}", flush=True)

    eval_dir = os.path.join(args.output_dir, f"atom_{atom_idx}_alpha_{args.alpha:.0e}")
    os.makedirs(eval_dir, exist_ok=True)

    # Create steered adapter
    print("\nCreating steered adapter...", flush=True)
    adapter_dir = os.path.join(eval_dir, "steered_adapter")
    create_steered_adapter(args.atom_path, args.alpha, CLEAN_ADAPTER, adapter_dir)

    # Eval baseline
    print("\nEval baseline (clean)...", flush=True)
    kill_gpu()
    time.sleep(3)
    proc = start_vllm("clean", CLEAN_ADAPTER, BASE_MODEL, PYTHON)
    baseline_results = None
    if proc:
        baseline_results = asyncio.run(eval_model("clean", GENERIC_EVAL_QS))
        proc.kill(); proc.wait()

    # Eval steered
    print("\nEval steered...", flush=True)
    kill_gpu()
    time.sleep(3)
    proc = start_vllm("steered", adapter_dir, BASE_MODEL, PYTHON)
    steered_results = None
    if proc:
        steered_results = asyncio.run(eval_model("steered", GENERIC_EVAL_QS))
        proc.kill(); proc.wait()
    kill_gpu()

    # Compare
    output = {
        "atom_idx": atom_idx, "coherence": coherence, "keywords": keywords,
        "alpha": args.alpha, "baseline": baseline_results, "steered": steered_results,
    }
    with open(os.path.join(eval_dir, "results.json"), "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*100}")
    print(f"ATOM {atom_idx} (coherence={coherence:.3f}, alpha={args.alpha:.0e})")
    print(f"Keywords: {', '.join(keywords[:10])}")
    print(f"{'='*100}")
    if baseline_results and steered_results:
        for b, s in zip(baseline_results, steered_results):
            print(f"\nQ: {b['q']}")
            print(f"  Baseline: {b['a'][:100]}")
            print(f"  Steered:  {s['a'][:100]}")
            if b['a'][:80] != s['a'][:80]:
                print(f"  *** CHANGED ***")

    print(f"\nResults saved to {eval_dir}/results.json", flush=True)


if __name__ == "__main__":
    main()
