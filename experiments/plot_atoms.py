"""Generate a publication-quality scatter plot of gradient atoms.

Usage:
    python experiments/plot_atoms.py
    python experiments/plot_atoms.py --method tsne --top_n 20
"""
from __future__ import annotations
import argparse, os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

from gradient_atoms.plotting import load_atom_data, embed_2d, plot_atoms

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default=os.path.join(PROJECT_ROOT, "results_alpha01"))
    parser.add_argument("--method", choices=["tsne", "umap", "pca"], default="tsne")
    parser.add_argument("--top_n", type=int, default=500)
    parser.add_argument("--output", default=os.path.join(PROJECT_ROOT, "figures", "atoms_scatter.png"))
    parser.add_argument("--perplexity", type=int, default=30)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    chars, D = load_atom_data(args.results_dir)

    if D is None:
        print("No dictionary matrix found (atoms.pt). Using coherence vs n_active as axes.")
        coords = np.array([[a["n_active"], a["coherence"]] for a in chars])
        plot_atoms(chars, coords, top_n=args.top_n, output_path=args.output,
                   title="Gradient Atoms: Coherence vs Sparsity")
    else:
        coords = embed_2d(D, method=args.method, perplexity=args.perplexity)
        plot_atoms(chars, coords, top_n=args.top_n, output_path=args.output)


if __name__ == "__main__":
    main()
