import argparse
from configs.model_config import ModelConfig
from configs.dataset_config import DatasetConfig

def parse_args():
    parser = argparse.ArgumentParser(description="LLM Benchmark Tool")
    
    # ===== 数据集参数 =====
    dataset_group = parser.add_argument_group("Dataset Config")
    dataset_group.add_argument("--dataset", type=str, required=True,
                             help="Dataset name (e.g., squad, xsum)")
    dataset_group.add_argument("--split", type=str, default="test",
                             help="Dataset split to use")
    dataset_group.add_argument("--prompt_key", type=str, default="default",
                             help="Prompt key to use")
    dataset_group.add_argument("--language", type=str, default="en",
                             help="Language split to use")
    dataset_group.add_argument("--max_samples", type=int, default=None,
                             help="Limit number of samples")
    dataset_group.add_argument("--sample_ratio", type=float, default=None,
                            help="Limit number of samples")

    # ===== 模型参数 =====
    model_group = parser.add_argument_group("Model Config")
    model_group.add_argument("--model_type", type=str, required=True,
                           choices=["hf", "vllm", "api"])
    model_group.add_argument("--model_path", type=str, required=True,
                           help="Path to model checkpoint")
    model_group.add_argument("--n_samples", type=int, default=1)
    model_group.add_argument("--timeout", type=int, default=600)

    model_group.add_argument("--dtype", type=str, default="bfloat16")
    model_group.add_argument("--workers", type=int, default=500)
    model_group.add_argument("--tensor_parallel_size", type=int, default=1)
    model_group.add_argument("--max_retries", type=int, default=30)
    model_group.add_argument("--api_url", type=str, default="http://localhost:8080/v1/completions")

    # ===== 生成参数 =====
    gen_group = parser.add_argument_group("Generation Config")
    gen_group.add_argument("--max_new_tokens", type=int, default=2048)
    gen_group.add_argument("--temperature", type=float, default=0)
    gen_group.add_argument("--do_sample", action="store_true")
    gen_group.add_argument("--top_p", type=float, default=None)
    gen_group.add_argument("--top_k", type=int, default=None)
    gen_group.add_argument("--stop_token_ids", type=int,nargs='+', default=[151645, 151643])
    # ===== 运行参数 =====
    runtime_group = parser.add_argument_group("Runtime Config")
    runtime_group.add_argument("--batch_size", type=int, default=8)
    runtime_group.add_argument("--output_dir", type=str, default="results")
    
    args = parser.parse_args()
    
    # 分别初始化配置对象
    model_config = ModelConfig(
        model_type=args.model_type,
        model_path=args.model_path,
        tensor_parallel_size = args.tensor_parallel_size,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=args.do_sample,
        top_p=args.top_p,
        top_k=args.top_k,
        batch_size=args.batch_size,
        api_url = args.api_url,
        timeout = args.timeout,
        workers = args.workers,
        max_retries = args.max_retries,
        n_samples = args.n_samples,
        stop_token_ids = args.stop_token_ids,
    )
    
    dataset_config = DatasetConfig(
        dataset=args.dataset,
        split=args.split,
        prompt_key=args.prompt_key,
        language=args.language,
        max_samples=args.max_samples,
        sample_ratio = args.sample_ratio,

    )
    
    return model_config, dataset_config, args.output_dir