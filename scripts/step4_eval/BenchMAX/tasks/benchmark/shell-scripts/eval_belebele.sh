

# 明确区分两类配置参数
# WORK_DIR="/mnt/nas/users/zhouhao.zh/new_project/chat_transfer/code/BenchMAX/tasks/benchmark"
# cd $WORK_DIR
# CUDA_VISIBLE_DEVICES=0 python main.py \
#   --dataset belebele \
#   --split test \
#   --prompt_key default \
#   --language "hu" \
#   \
#   --model_type api \
#   --model_path /mnt/nas/users/zhouhao.zh/LLMs/Qwen2.5-7B-Instruct \
#   --workers 200 \
#   \
#   --max_new_tokens 1024 \
#   --temperature 0 \
#   \
#   --batch_size 16 \
#   --output_dir results/test_api/belebele_qwen2.5_instruct_en
