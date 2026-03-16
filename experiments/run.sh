#!/bin/bash
# Run the full gradient atoms pipeline
#
# Step 1: Extract per-doc gradients (multi-GPU, ~3 min)
# Step 2: Dictionary learning (CPU, ~15 min)
# Step 3: Evaluate top atoms (GPU, ~10 min each)
#
# Set INFUSION_ROOT to point to the infusion repo if not in ~/infusion.

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

export INFUSION_ROOT="${INFUSION_ROOT:-$HOME/infusion}"
PYTHON="${INFUSION_ROOT}/.venv/bin/python"

echo "=========================================="
echo "Step 1: Extract per-doc gradients (8 GPUs)"
echo "=========================================="
torchrun --nproc_per_node=8 \
    "${SCRIPT_DIR}/extract_gradients.py" \
    --n_docs 5000 \
    --output_dir "${PROJECT_ROOT}/results"

echo ""
echo "=========================================="
echo "Step 2: Dictionary learning"
echo "=========================================="
$PYTHON "${SCRIPT_DIR}/learn_atoms.py" \
    --gradients_path "${PROJECT_ROOT}/results/gradients_all.pt" \
    --n_atoms 500 \
    --top_k_eigen 50 \
    --alpha 0.1 \
    --output_dir "${PROJECT_ROOT}/results"

echo ""
echo "=========================================="
echo "Step 3: Evaluate top atoms"
echo "=========================================="
for i in $(seq 0 4); do
    ATOM_FILE=$(ls "${PROJECT_ROOT}/results/steering_vectors/atom_"*.pt 2>/dev/null | head -n $((i+1)) | tail -1)
    if [ -n "$ATOM_FILE" ]; then
        echo "Evaluating: $ATOM_FILE"
        $PYTHON "${SCRIPT_DIR}/steer_atom.py" \
            --atom_path "$ATOM_FILE" \
            --alpha 1e-4 \
            --output_dir "${PROJECT_ROOT}/results/eval"
    fi
done

echo ""
echo "Done! Results in ${PROJECT_ROOT}/results/"
