# DeltaMoE: Data-Efficient Multilingual LLM Expansion

This repository contains the code for our ACL 2026 paper:

> **A Data-Efficient Path to Multilingual LLMs: Language Expansion via Post-training PARAM∆ Integration into Upcycled MoE**
>
> Hao Zhou, Tianhao Li, Zhijun Wang, Shuaijie She, Linjuan Wu, Hao-Ran Wei, Baosong Yang, Jiajun Chen, Shujian Huang
>
> National Key Laboratory for Novel Software Technology, Nanjing University | Tongyi Lab, Alibaba Group | Zhejiang University

## Overview

Expanding LLMs to new languages typically requires costly Continued Pre-Training (CPT) with massive data replay, followed by expensive alignment. Existing parameter-merging approaches face a critical trade-off: linear averaging preserves original capabilities but dilutes new language acquisition, while delta merging acquires new languages but causes catastrophic forgetting.

**DeltaMoE** resolves this conflict with a two-stage framework:

```
Dense Model
    │
    ▼  Step 1: Upcycling
Upcycled MoE (1 frozen original expert + N-1 new experts)
    │
    ▼  Step 2: CPT Training  (train only new experts + router)
CPT-enhanced MoE
    │
    ▼  Step 3: Delta Merge   new = moe + instruct_delta - base_delta
DeltaMoE (instruction-following MoE, no alignment data needed)
    │
    ▼  Step 4: Evaluation
Results on MGSM / IFEval / Belebele / M-MMLU / Flores
```

## Directory Structure

```
open_source/
├── README.md
├── transformers/           # Custom transformers with Qwen2MoELPR model
├── LLaMA-Factory/          # Custom LLaMA-Factory with moelpr finetuning support
├── lm-evaluation-harness/  # Custom lm-eval with EOS token fixes
└── scripts/
    ├── step1_upcycling/    # Convert dense → MoE
    ├── step2_train/        # CPT training with LLaMA-Factory
    ├── step3_delta_merge/  # Delta merge to transfer instruction-following
    └── step4_eval/         # Evaluation (5 tasks, BenchMAX + lm-eval)
        └── BenchMAX/       # Evaluation package (Belebele, M-MMLU, Flores, IFEval, MGSM)
```

## Installation

### 1. Install custom transformers

```bash
cd transformers
pip install -e .
```

Verify: `python -c "from transformers import Qwen2MoELPRForCausalLM; print('OK')`

### 2. Install custom LLaMA-Factory

```bash
cd LLaMA-Factory
pip install -e .
```

### 3. Install custom lm-evaluation-harness

```bash
cd lm-evaluation-harness
pip install -e .
```

### 4. Install BenchMAX evaluation dependencies

```bash
pip install -r scripts/step4_eval/BenchMAX/requirements.txt
```

---

## Step-by-Step Reproduction

### Step 1: Upcycle Dense Model to MoE

Edit paths in `scripts/step1_upcycling/run_upcycling.sh`, then:

```bash
bash scripts/step1_upcycling/run_upcycling.sh
```

Or directly:

```bash
python scripts/step1_upcycling/upcycling.py \
    --model_path /path/to/Qwen2.5-7B \
    --num_experts 4 \
    --output_path /path/to/output/qwen2.5_7B_4experts
```

This converts the dense model's FFN layers into MoE with `num_experts` experts (all initialized as copies of the original FFN).

### Step 2: CPT Training

First, register your multilingual pretraining data in `LLaMA-Factory/data/dataset_info.json`. Then edit `scripts/step2_train/train.sh` and set your paths:

```bash
bash scripts/step2_train/train.sh
```

Key training arguments:
- `--finetuning_type moelpr` — enables MoE LPR training mode
- `--topk 2` — top-2 expert routing
- `--lpr_stage 3` — LPR training stage (stage 3 = standard top-k routing)
- `--aux_loss_coef 0.01` — load balancing loss coefficient

DeepSpeed configs are in `scripts/step2_train/ds_config/`. We use ZeRO-3 for 7B models.

**Data format**: Standard LLaMA-Factory pretraining format (plain text JSONL). Refer to `LLaMA-Factory/data/README.md` for dataset registration.

### Step 3: Delta Merge

Edit paths in `scripts/step3_delta_merge/run_delta_merge.sh`, then:

```bash
bash scripts/step3_delta_merge/run_delta_merge.sh
```

Or directly:

```bash
python scripts/step3_delta_merge/delta_merge.py \
    --base_model_path /path/to/Qwen2.5-7B \
    --cpt_model_path /path/to/output/qwen2.5_7B_4experts_cpt \
    --instruct_model_path /path/to/Qwen2.5-7B-Instruct \
    --output_path /path/to/output/deltamoe_merged
```

The merge formula: `new_param = moe_param + instruct_param - base_param`
- Non-MoE layers (attention, norms, embeddings): delta applied directly
- MoE expert layers: instruct delta broadcast uniformly across all experts
- Router gate weights: preserved from CPT model unchanged

### Step 4: Evaluation

Edit paths in `scripts/step4_eval/eval.sh`, then run per language:

```bash
# Evaluate on Hungarian
bash scripts/step4_eval/eval.sh --plm /path/to/deltamoe_merged \
    --savedir /path/to/results --lang hu --device 0

# Evaluate on all paper languages
for lang in hu bn sr en zh es fr; do
    bash scripts/step4_eval/eval.sh --plm /path/to/deltamoe_merged \
        --savedir /path/to/results --lang $lang --device 0
done
```

Print aggregated results:

```bash
python scripts/step4_eval/print_results.py \
    --result_path /path/to/results/deltamoe_merged \
    --expand_lang hu bn sr \
    --original_lang en zh es fr
```

---

## Supported Models

The paper evaluates on:
- **Qwen2.5-7B** / **Qwen2.5-7B-Instruct** (primary experiments)
- **LLaMA-3.1-8B** / **LLaMA-3.1-8B-Instruct** (cross-model generalization)

For LLaMA-3.1, use the `llama_moelpr` model type. Upcycling for LLaMA follows the same procedure — the custom transformers library handles both `qwen2_moelpr` and `llama_moelpr` architectures.


