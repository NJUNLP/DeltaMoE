
eval "$(/mnt/nas/users/zichen.shx/miniconda3/bin/conda shell.bash hook)" 
conda activate /mnt/nas/users/zhouhao.zh/envs/verl_qwen3

export HF_ENDPOINT=https://hf-mirror.com
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_MODELSCOPE=False
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

WORK_DIR="/mnt/nas/users/zhouhao.zh/new_project/chat_transfer/code/BenchMAX/tasks/benchmark"
cd $WORK_DIR
python main.py \
  --dataset aime \
  --split train \
  --prompt_key "prompt" \
  --language "en" \
  \
  --model_type vllm \
  --model_path /mnt/nas_intern/users/zhouhao.zh/new_project/chat_transfer/ckpt/qwen3/merged/tian/tian_qwen3_base_4b_hu10b \
  --tensor_parallel_size 4 \
  --n_samples 16 \
  \
  --max_new_tokens 32768 \
  --temperature 0.6 \
  --top_k 20 \
  --top_p 0.95 \
  \
  --batch_size 16 \
  --output_dir /mnt/nas/users/zhouhao.zh/new_project/chat_transfer/hf_instruct_res/Qwen3/merge/tian/tian_qwen3_base_4b_hu10b/aime




eval "$(/mnt/nas/users/zichen.shx/miniconda3/bin/conda shell.bash hook)" 
conda activate /mnt/nas/users/zhouhao.zh/envs/verl_qwen3

export HF_ENDPOINT=https://hf-mirror.com
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_MODELSCOPE=False
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

WORK_DIR="/mnt/nas/users/zhouhao.zh/new_project/chat_transfer/code/BenchMAX/tasks/benchmark"
cd $WORK_DIR
python main.py \
  --dataset aime \
  --split train \
  --prompt_key "prompt" \
  --language "en" \
  \
  --model_type vllm \
  --model_path /mnt/nas/users/zhouhao.zh/LLMs/Qwen3/Qwen3-4B \
  --tensor_parallel_size 4 \
  --n_samples 16 \
  \
  --max_new_tokens 32768 \
  --temperature 0.6 \
  --top_k 20 \
  --top_p 0.95 \
  \
  --batch_size 16 \
  --output_dir /mnt/nas/users/zhouhao.zh/new_project/chat_transfer/hf_instruct_res/Qwen3/Qwen3-4B/aime


python /mnt/nas/users/zhouhao.zh/omni/code/test.py

# --model_path /mnt/nas/users/zhouhao.zh/LLMs/Qwen3/Qwen3-4B \


# --model_path /mnt/nas/users/zhouhao.zh/LLMs/Qwen3/Qwen3-4B \
