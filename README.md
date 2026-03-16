# Gradient Atoms

**Unsupervised discovery and steering of model behaviors via sparse decomposition of training gradients.**

[Paper](PAPER.tex) | [Blog post](less_wrong.md)

## Quick Start

```bash
uv sync

# Step 1: Extract per-document gradients (8 GPUs, ~170s)
torchrun --nproc_per_node=8 experiments/extract_gradients.py --n_docs 5000

# Step 2-4: Project, decompose, and characterise atoms (~25 min CPU)
python experiments/learn_atoms.py

# Step 5: Evaluate steering
python experiments/steer_behavioral_atoms.py
```

Requires Phase 1 outputs (LoRA adapter + EKFAC factors) — see `experiments/upstream/`.

## Citation

```bibtex
@inproceedings{gradient-atoms-2026,
  title     = {Gradient Atoms: Unsupervised Discovery of Model Behaviors
               via Sparse Decomposition of Training Gradients},
  year      = {2026},
  author    = {J Rosser}
}
```

## License

Apache 2.0
