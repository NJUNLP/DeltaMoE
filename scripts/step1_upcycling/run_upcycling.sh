#!/bin/bash
# ============================================================
# Step 1: Upcycle a dense model to a Qwen2MoELPR MoE model
# ============================================================

# ---- Configure paths below ----
MODEL_PATH="/path/to/Qwen2.5-7B"          # Base dense model
OUTPUT_PATH="/path/to/output/qwen2.5_7B_4experts"  # Output MoE checkpoint
NUM_EXPERTS=4                              # Number of experts (paper uses 4)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Install custom transformers first (see README)
python "${SCRIPT_DIR}/upcycling.py" \
    --model_path "${MODEL_PATH}" \
    --num_experts "${NUM_EXPERTS}" \
    --output_path "${OUTPUT_PATH}"

echo "Upcycling done. Output: ${OUTPUT_PATH}"
