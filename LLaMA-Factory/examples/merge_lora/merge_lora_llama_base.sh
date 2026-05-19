eval "$(/mnt/nas/users/zichen.shx/miniconda3/bin/conda shell.bash hook)" 
conda activate /mnt/nas/users/zhouhao.zh/envs/llama_llamfactory_moelpr


MODEL_DIR="/mnt/nas/users/zhouhao.zh/LLMs/meta-llama/Meta-Llama-3.1-8B"
ADAPTER_DIR="/mnt/nas/users/zhouhao.zh/new_project/hf_ckpt/moelpr/llama3/dense/Meta-Llama-3.1-8B_hu_3B_bn_3B_sr_3B_2048_lr1e-4_lora_64"
python /mnt/nas/users/zhouhao.zh/new_project/chat_transfer/code/MoE-LPR/LLaMA-Factory/examples/merge_lora/merge_adapter.py \
    --model_path $MODEL_DIR \
    --adapter_name_or_path  $ADAPTER_DIR \
    --output_path "${ADAPTER_DIR}_dense_merged"


MODEL_DIR="/mnt/nas/users/zhouhao.zh/LLMs/meta-llama/Meta-Llama-3.1-8B-Instruct"
ADAPTER_DIR="/mnt/nas/users/zhouhao.zh/new_project/hf_ckpt/moelpr/llama3/dense/Meta-Llama-3.1-8B_instruct_hu_3B_bn_3B_sr_3B_2048_lr1e-4_lora_64"
python /mnt/nas/users/zhouhao.zh/new_project/chat_transfer/code/MoE-LPR/LLaMA-Factory/examples/merge_lora/merge_adapter.py \
    --model_path $MODEL_DIR \
    --adapter_name_or_path  $ADAPTER_DIR \
    --output_path "${ADAPTER_DIR}_dense_merged"