#!/bin/bash
# ============================================================
# Step 2: CPT Training — train the upcycled MoE model
# Uses LLaMA-Factory with custom moelpr finetuning_type
# ============================================================

# ---- Configure paths below ----
LLAMA_FACTORY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../LLaMA-Factory" && pwd)"

# Model paths
UPCYCLED_MODEL_PATH="/path/to/output/qwen2.5_7B_4experts"  # Output from Step 1
OUTPUT_DIR="/path/to/output/qwen2.5_7B_4experts_cpt"        # CPT checkpoint output

# Data: comma-separated LLaMA-Factory dataset names
# Register your datasets in LLaMA-Factory/data/dataset_info.json first
DATASETS="your_lang1_data,your_lang2_data"

# DeepSpeed config (z3 recommended for 7B+)
DEEPSPEED_CONFIG="${LLAMA_FACTORY_DIR}/examples/deepspeed/ds_z3_config.json"

# Distributed training settings (set these for multi-node or use defaults for single-node)
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"29500"}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
WORLD_SIZE=${WORLD_SIZE:-1}
RANK=${RANK:-0}

# ---- Training config ----
export NCCL_BLOCKING_WAIT=1
export NCCL_COMM_TIMEOUT=360000
export WANDB_MODE=offline
export WANDB_DISABLED=true

cd "${LLAMA_FACTORY_DIR}"

python -m torch.distributed.run \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    --nproc_per_node "${NPROC_PER_NODE}" \
    --nnodes "${WORLD_SIZE}" \
    --node_rank "${RANK}" \
    src/train.py \
    --deepspeed "${DEEPSPEED_CONFIG}" \
    --stage pt \
    --model_name_or_path "${UPCYCLED_MODEL_PATH}" \
    --do_train \
    --finetuning_type moelpr \
    --topk 2 \
    --lpr_stage 3 \
    --aux_loss_coef 0.01 \
    --dataset "${DATASETS}" \
    --cutoff_len 2048 \
    --overwrite_cache \
    --preprocessing_num_workers 32 \
    --packing true \
    --output_dir "${OUTPUT_DIR}" \
    --logging_steps 10 \
    --save_steps 4000 \
    --plot_loss \
    --save_only_model \
    --overwrite_output_dir \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5.0e-5 \
    --num_train_epochs 1 \
    --lr_scheduler_type cosine \
    --bf16 true \
    --ddp_timeout 180000000

echo "CPT training done. Output: ${OUTPUT_DIR}"
