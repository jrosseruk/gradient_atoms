"""Step 2: EKFAC projection + sparse dictionary learning on per-doc gradients.

Loads the gradient matrix from Step 1, projects into the EKFAC eigenbasis,
runs sparse dictionary learning, and characterises each atom.

Usage:
    python experiments/learn_atoms.py --n_atoms 500 --top_k_eigen 50
    python experiments/learn_atoms.py --n_atoms 200 --skip_projection
"""
from __future__ import annotations
import argparse, json, os, sys, time
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
INFUSION_ROOT = os.environ.get("INFUSION_ROOT", os.path.expanduser("~/infusion"))
UK_EXPERIMENTS = os.path.join(INFUSION_ROOT, "experiments_infusion_uk")
sys.path.insert(0, UK_EXPERIMENTS)
sys.path.insert(0, INFUSION_ROOT)

FACTORS_DIR = os.path.join(
    UK_EXPERIMENTS, "attribute", "results_v4",
    "infusion_uk_ekfac", "factors_infusion_uk_factors")
DATA_REPO = "jrosseruk/subl-learn-data"

from gradient_atoms.projection import load_ekfac_eigen, project_gradients_ekfac, unproject_atom
from gradient_atoms.dictionary import run_dictionary_learning, characterise_atoms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gradients_path", default=os.path.join(PROJECT_ROOT, "results", "gradients_all.pt"))
    parser.add_argument("--n_atoms", type=int, default=500)
    parser.add_argument("--top_k_eigen", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--max_iter", type=int, default=100)
    parser.add_argument("--skip_projection", action="store_true")
    parser.add_argument("--output_dir", default=os.path.join(PROJECT_ROOT, "results"))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading gradients...", flush=True)
    data = torch.load(args.gradients_path, weights_only=True)
    gradients = data["gradients"]
    lora_names = data["lora_names"]
    n_docs = gradients.shape[0]
    d = gradients.shape[1]
    print(f"  Loaded: {n_docs} docs, {d} params", flush=True)

    print("Loading training docs for labelling...", flush=True)
    sys.path.insert(0, os.path.join(UK_EXPERIMENTS, "attribute"))
    from compute_ekfac_v4 import load_clean_training_data
    docs = load_clean_training_data(DATA_REPO, n_docs)

    if args.skip_projection:
        print("Skipping EKFAC projection (using raw gradients)...", flush=True)
        G_proj = gradients
        module_info = None
        ekfac_modules = None
    else:
        print("Loading EKFAC factors...", flush=True)
        ekfac_modules = load_ekfac_eigen(FACTORS_DIR)
        print(f"  Loaded {len(ekfac_modules)} modules", flush=True)

        print("Projecting gradients into EKFAC eigenbasis...", flush=True)
        t0 = time.time()
        G_proj, module_info = project_gradients_ekfac(
            gradients, lora_names, ekfac_modules, args.top_k_eigen)
        print(f"  Projection done in {time.time()-t0:.0f}s", flush=True)

    print(f"\nRunning dictionary learning ({args.n_atoms} atoms)...", flush=True)
    D, A, grad_norms = run_dictionary_learning(
        G_proj, n_atoms=args.n_atoms, alpha=args.alpha, max_iter=args.max_iter)

    print("\nCharacterising atoms...", flush=True)
    atom_info = characterise_atoms(D, A, gradients, docs, args.n_atoms)
    atom_info.sort(key=lambda x: -x["coherence"])

    print("\nSaving results...", flush=True)
    torch.save({
        "dictionary": torch.tensor(D, dtype=torch.float32),
        "coefficients": torch.tensor(A, dtype=torch.float32),
        "grad_norms": torch.tensor(grad_norms, dtype=torch.float32),
        "module_info": module_info,
        "lora_names": lora_names,
        "n_atoms": args.n_atoms,
        "top_k_eigen": args.top_k_eigen,
        "alpha": args.alpha,
    }, os.path.join(args.output_dir, "atoms.pt"))

    with open(os.path.join(args.output_dir, "atom_characterisations.json"), "w") as f:
        json.dump(atom_info, f, indent=2)

    print(f"\n{'='*100}")
    print(f"TOP 50 ATOMS BY COHERENCE")
    print(f"{'='*100}")
    print(f"{'Rank':>4} {'Atom':>5} {'Coher':>7} {'nActive':>8} {'MeanCoeff':>10}  Keywords")
    print("-" * 100)
    for i, a in enumerate(atom_info[:50]):
        kw = ", ".join(a["keywords"][:8])
        print(f"{i+1:>4} {a['atom_idx']:>5} {a['coherence']:>7.3f} "
              f"{a['n_active']:>8} {a['mean_coeff']:>10.4f}  {kw}")

    if ekfac_modules is not None and module_info is not None:
        print("\nUnprojecting top coherent atoms to steering vectors...", flush=True)
        steering_dir = os.path.join(args.output_dir, "steering_vectors")
        os.makedirs(steering_dir, exist_ok=True)

        for i, a in enumerate(atom_info[:50]):
            if a["coherence"] < 0.5:
                break
            atom_vec = D[a["atom_idx"]]
            sv = unproject_atom(atom_vec, module_info, ekfac_modules, d)
            torch.save({
                "v_flat": sv,
                "atom_idx": a["atom_idx"],
                "coherence": a["coherence"],
                "n_active": a["n_active"],
                "keywords": a["keywords"],
            }, os.path.join(steering_dir, f"atom_{a['atom_idx']:04d}.pt"))

        print(f"  Saved steering vectors for top coherent atoms", flush=True)

    print("\nDone!", flush=True)


if __name__ == "__main__":
    main()
