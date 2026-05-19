import time
import logging
import requests
import tenacity
import argparse
from functools import partial
from multiprocessing import Pool, Manager
import tqdm  # 可选：用于进度条

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 示例语言字典（根据实际情况替换）
langdic = {'en': 'English', 'es': 'Spanish', 'fr': 'French'}

# VLLM服务器配置
VLLM_SERVER_URL = "http://localhost:8000/generate"  # 替换为实际API地址
tokenizer = None  # 需替换为实际的tokenizer初始化

def init_tokenizer():
    """初始化tokenizer（根据实际情况实现）"""
    global tokenizer
    # 示例：tokenizer = AutoTokenizer.from_pretrained("your-model")
    pass

@tenacity.retry(stop=tenacity.stop_after_attempt(3), wait=tenacity.wait_fixed(2))
def api_call(task):
    """带重试机制的API调用（适配多进程版本）"""
    prompt, args = task
    template = f"Translate the following text into {langdic[args.lang]}. Please put your translation between <translation> your translation </translation>. \n\n[Source Text]: {prompt}"
    messages = [{"role": "user", "content": template}]
    
    # 实际应用中需替换为真实的tokenizer使用方式
    if tokenizer is None:
        init_tokenizer()
    query = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True, 
        enable_thinking=False
    )
    
    request_data = {
        "prompt": query,
        "max_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "n": 1
    }
    
    try:
        response = requests.post(
            VLLM_SERVER_URL,
            json=request_data,
            timeout=args.timeout
        )
        response.raise_for_status()
        return response.json()['choices'][0]['text']
    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed: {str(e)}")
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
    return None

def main():
    # 参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True, help="输入文本文件路径")
    parser.add_argument("--output-file", type=str, required=True, help="输出结果文件路径")
    parser.add_argument("--lang", type=str, default='en', choices=langdic.keys(), help="目标语言")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="最大生成token数")
    parser.add_argument("--temperature", type=float, default=0.7, help="采样温度")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p采样参数")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k采样参数")
    parser.add_argument("--timeout", type=int, default=30, help="API超时时间(秒)")
    parser.add_argument("--workers", type=int, default=1024, help="并发进程数(1-1024)")
    args = parser.parse_args()

    # 验证工作进程数
    workers = max(1, min(args.workers, 1024))
    logging.info(f"Starting processing with {workers} workers")

    # 读取输入数据
    with open(args.input_file, 'r') as f:
        prompts = [line.strip() for line in f if line.strip()]
    
    if not prompts:
        logging.error("No valid prompts found in input file")
        return

    # 创建任务队列 (prompt, args) 元组
    tasks = [(prompt, args) for prompt in prompts]

    # 使用进程池处理
    results = []
    try:
        with Pool(processes=workers) as pool:
            # 使用imap_unordered获取结果（更快）
            # 添加tqdm进度条（可选）
            for result in tqdm.tqdm(
                pool.imap_unordered(partial(api_call), tasks),
                total=len(tasks),
                desc="Translating"
            ):
                results.append(result)
    except KeyboardInterrupt:
        logging.warning("Process interrupted by user")
        pool.terminate()
        pool.join()
        return

    # 写入结果
    with open(args.output_file, 'w') as f:
        for res in results:
            f.write(f"{res or 'ERROR'}\n")

    logging.info(f"Process completed. Results saved to {args.output_file}")

if __name__ == "__main__":
    # Windows系统需要添加freeze_support()
    # from multiprocessing import freeze_support
    # freeze_support()
    main()