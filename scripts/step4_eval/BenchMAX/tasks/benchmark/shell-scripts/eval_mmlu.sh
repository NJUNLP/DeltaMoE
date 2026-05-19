

# 明确区分两类配置参数
WORK_DIR="/mnt/nas/users/zhouhao.zh/new_project/chat_transfer/code/BenchMAX/tasks/benchmark"
cd $WORK_DIR
CUDA_VISIBLE_DEVICES=0 python main.py \
  --dataset m_mmlu \
  --split test \
  --language "hu" \
  --sample_ratio 0.1 \
  \
  --model_type hf \
  --model_path /mnt/nas/users/zhouhao.zh/LLMs/Qwen2.5-7B-Instruct \
  \
  --max_new_tokens 1024 \
  --temperature 0 \
  \
  --batch_size 50 \
  --output_dir results/mmlu/qwen2.5_instruct_en
