#!/bin/bash
# ============================================================
# Step 4: Evaluate a DeltaMoE (or any instruction-tuned) model
# Covers 5 tasks: MGSM, IFEval, Belebele, M-MMLU, Flores
# ============================================================

# ---- Configure paths below ----
PLM="/path/to/your/model"           # Model to evaluate
SAVEDIR="/path/to/results"          # Directory to store results
LANG="hu"                           # Language code (e.g., hu, bn, sr, en, zh, fr, es)
DEVICE="0"                          # CUDA device index

# Batch sizes (reduce if OOM)
XGSM_BSZ=120
XIFEVAL_BSZ=120
BELEBELE_BSZ=120
MMLU_BSZ=50
FLORES_BSZ=120

# ---- Paths (relative to this script) ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMAX_DIR="${SCRIPT_DIR}/BenchMAX"
LM_EVAL_DIR="${SCRIPT_DIR}/../../lm-evaluation-harness"

# Parse --key value arguments
while [ $# -gt 0 ]; do
    if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare "$param"="$2"
    fi
    shift
done

export HF_ENDPOINT=https://hf-mirror.com
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_MODELSCOPE=False

PLMNAME=$(basename "$PLM")
echo "Evaluating: ${PLMNAME}, lang: ${LANG}"

# ---- MGSM ----
RESULT_PATH="${SAVEDIR}/${PLMNAME}/xgsm/${LANG}"
mkdir -p "${RESULT_PATH}"
if [ -z "$(find "${RESULT_PATH}" -type f -name "*.json" | head -n 1)" ]; then
    echo "[MGSM] Starting evaluation..."
    cd "${BENCHMAX_DIR}"
    lm-eval -m hf --model_args pretrained=${PLM},dtype=bfloat16 \
        --tasks xmgsm_native_cot_${LANG} \
        --batch_size ${XGSM_BSZ} --apply_chat_template \
        --include_path tasks/mgsm --log_samples \
        -o "${RESULT_PATH}" --device cuda:${DEVICE}
fi

# ---- IFEval ----
RESULT_PATH="${SAVEDIR}/${PLMNAME}/xifeval/${LANG}"
mkdir -p "${RESULT_PATH}"
if [ -z "$(find "${RESULT_PATH}" -type f -name "*.json" | head -n 1)" ]; then
    echo "[IFEval] Starting evaluation..."
    cd "${BENCHMAX_DIR}"
    lm-eval -m hf --model_args pretrained=${PLM},dtype=bfloat16 \
        --tasks xifeval_${LANG} \
        --batch_size ${XIFEVAL_BSZ} --apply_chat_template \
        --include_path tasks/ifeval --log_samples \
        -o "${RESULT_PATH}" --device cuda:${DEVICE}
fi

# ---- Belebele language code mapping ----
declare -A langMap
langMap["el"]="ell_Grek"; langMap["tr"]="tur_Latn"; langMap["hu"]="hun_Latn"
langMap["en"]="eng_Latn"; langMap["es"]="spa_Latn"; langMap["ar"]="arb_Arab"
langMap["ru"]="rus_Cyrl"; langMap["de"]="deu_Latn"; langMap["zh"]="zho_Hans"
langMap["fr"]="fra_Latn"; langMap["pt"]="por_Latn"; langMap["ja"]="jpn_Jpan"
langMap["bn"]="ben_Beng"; langMap["sr"]="srp_Cyrl"; langMap["cs"]="ces_Latn"
langMap["vi"]="vie_Latn"

if [ "$LANG" = "bn" ] || [ "$LANG" = "te" ] || [ "$LANG" = "el" ] || [ "$LANG" = "cs" ]; then
    MAX_TOKENS=2048
else
    MAX_TOKENS=1024
fi

# ---- Belebele ----
RESULT_PATH="${SAVEDIR}/${PLMNAME}/belebele/${LANG}"
mkdir -p "${RESULT_PATH}"
if [ -z "$(find "${RESULT_PATH}" -type f -name "*.json" | head -n 1)" ]; then
    echo "[Belebele] Starting evaluation..."
    CUDA_VISIBLE_DEVICES=${DEVICE} python "${BENCHMAX_DIR}/tasks/benchmark/main.py" \
        --dataset belebele --split test --language ${LANG} \
        --model_type hf --model_path "${PLM}" \
        --max_new_tokens ${MAX_TOKENS} --temperature 0 \
        --batch_size ${BELEBELE_BSZ} --output_dir "${RESULT_PATH}"
fi

# ---- M-MMLU ----
RESULT_PATH="${SAVEDIR}/${PLMNAME}/mmlu/${LANG}"
mkdir -p "${RESULT_PATH}"
if [ -z "$(find "${RESULT_PATH}" -type f -name "*.json" | head -n 1)" ]; then
    echo "[M-MMLU] Starting evaluation..."
    CUDA_VISIBLE_DEVICES=${DEVICE} python "${BENCHMAX_DIR}/tasks/benchmark/main.py" \
        --dataset m_mmlu --split test --language ${LANG} --sample_ratio 0.1 \
        --model_type hf --model_path "${PLM}" \
        --max_new_tokens ${MAX_TOKENS} --temperature 0 \
        --batch_size ${MMLU_BSZ} --output_dir "${RESULT_PATH}"
fi

# ---- Flores (translation) ----
RESULT_PATH="${SAVEDIR}/${PLMNAME}/flores"
mkdir -p "${RESULT_PATH}"
if [ -z "$(find "${RESULT_PATH}" -type f -name "*${LANG}-zh.json" | head -n 1)" ]; then
    echo "[Flores] Starting translation evaluation..."
    cd "${BENCHMAX_DIR}/tasks/translation"
    CUDA_VISIBLE_DEVICES=${DEVICE} python generate_translation_hf.py \
        -s en,zh -t ${LANG} --task-name flores --model-name "${PLM}" \
        --infer-backend vllm --max-tokens 512 --output-dir "${RESULT_PATH}" \
        --batch-size ${FLORES_BSZ}
    CUDA_VISIBLE_DEVICES=${DEVICE} python generate_translation_hf.py \
        -s ${LANG} -t en,zh --task-name flores --model-name "${PLM}" \
        --infer-backend vllm --max-tokens 512 --output-dir "${RESULT_PATH}" \
        --batch-size ${FLORES_BSZ}
    CUDA_VISIBLE_DEVICES=${DEVICE} python evaluate_translation.py \
        -s en,zh -t ${LANG} --task-name flores --model-name "${PLM}" \
        --metrics xComet --output-dir "${RESULT_PATH}"
    CUDA_VISIBLE_DEVICES=${DEVICE} python evaluate_translation.py \
        -s ${LANG} -t en,zh --task-name flores --model-name "${PLM}" \
        --metrics xComet --output-dir "${RESULT_PATH}"
fi

echo "Evaluation complete. Results in: ${SAVEDIR}/${PLMNAME}"
