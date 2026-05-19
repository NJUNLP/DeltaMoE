#!/bin/bash
# ============================================================
# Step 3: Delta Merge — apply instruction-tuning delta to MoE
# Formula: new_param = moe_param + instruct_param - base_param
# ============================================================

# ---- Configure paths below ----
BASE_MODEL_PATH="/path/to/Qwen2.5-7B"                # Original dense base model
CPT_MODEL_PATH="/path/to/output/qwen2.5_7B_4experts_cpt"  # Output from Step 2
INSTRUCT_MODEL_PATH="/path/to/Qwen2.5-7B-Instruct"   # Instruction-tuned dense model
OUTPUT_PATH="/path/to/output/deltamoe_merged"         # Final DeltaMoE model

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "${SCRIPT_DIR}/delta_merge.py" \
    --base_model_path "${BASE_MODEL_PATH}" \
    --cpt_model_path "${CPT_MODEL_PATH}" \
    --instruct_model_path "${INSTRUCT_MODEL_PATH}" \
    --output_path "${OUTPUT_PATH}"

echo "Delta merge done. Output: ${OUTPUT_PATH}"
