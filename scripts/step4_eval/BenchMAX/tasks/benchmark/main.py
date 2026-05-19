from utils.cli import parse_args
from runners.benchmark_runner import run_benchmark

"""
WORK_DIR="/mnt/nas/users/zhouhao.zh/new_project/chat_transfer/code/BenchMAX/tasks/benchmark"
cd $WORK_DIR
python main.py \
  --dataset m_mmlu \
  --split test \
  --language "hu" \
  \
  --model_type vllm \
  --model_path /mnt/nas/users/zhouhao.zh/LLMs/Qwen2.5-7B-Instruct \
  \
  --max_new_tokens 1024 \
  --temperature 0 \
  \
  --batch_size 16 \
  --output_dir results/mmlu
"""

if __name__ == "__main__":
    # 解析命令行参数 → 现在返回三个对象
    model_config, dataset_config, output_dir = parse_args()
    
    # 运行评测
    results = run_benchmark(
        model_config=model_config,
        dataset_config=dataset_config,
        output_dir=output_dir
    )
    