import os
import concurrent.futures
import tqdm
from vllm import LLM, SamplingParams
import openai
import requests
import time
import logging
import json
from tenacity import retry, stop_after_attempt, wait_exponential
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
"""
 python generate_translation.py -s en,zh -t hu  --task-name flores --model-name /mnt/nas/users/zhouhao.zh/LLMs/Qwen2.5-7B-Instruct --infer-backend localapi --max-tokens 512 --output-dir /mnt/nas/users/zhouhao.zh/test_code --base-url http://localhost:8080/v1/completions
"""
def build_model(
    model_name: str,
    infer_backend: str,
    tokenizer: str = None,
    tensor_parallel_size: int = 1,
    trust_remote_code: bool = False,
    base_url: str = None,
    num_concurrent: int = 1,
    **kwargs
):
    match infer_backend:
        case "vllm":
            return VLLMCausalLLM(
                model_path=model_name,
                tokenizer=tokenizer,
                tensor_parallel_size=tensor_parallel_size,
                trust_remote_code=trust_remote_code,
                **kwargs
            )
        case "openai_api":
            api_key = os.getenv("OPENAI_API_KEY", None)
            if api_key is None:
                raise ValueError(
                    "API key not found. Please set the `OPENAI_API_KEY` environment variable."
                )
            return OpenAILLM(model_name=model_name, base_url=base_url, api_key=api_key, num_concurrent=num_concurrent, **kwargs)
        case "localapi":
            # 添加对本地API的支持
            if not base_url:
                raise ValueError("base_url is required for localapi backend")
            return LocalAPILLM(
                api_url=base_url,
                model_name=model_name,
                num_concurrent=num_concurrent,
                **kwargs
            )
        case _:
            raise ValueError(f"Unknown infer_backend: {infer_backend}")



class VLLMCausalLLM:
    def __init__(self, model_path: str, tokenizer: str = None, tensor_parallel_size: int = 1, trust_remote_code: bool = False, **kwargs):
        if tensor_parallel_size is None or tensor_parallel_size < 0:
            tensor_parallel_size = torch.cuda.device_count()
        self.model = LLM(
            model=model_path,
            tokenizer=tokenizer,
            tensor_parallel_size=tensor_parallel_size,
            # dtype="bfloat16",
            dtype="auto",
            trust_remote_code=trust_remote_code,
            **kwargs
        )

    def chat(self, messages, temperature=0.0, max_tokens=512):
        sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        response = self.model.chat(messages, sampling_params=sampling_params)
        outputs = [r.outputs[0].text.strip() for r in response]
        return outputs

    def generate(self, prompts, temperature=0.0, max_tokens=512):
        sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        response = self.model.generate(prompts, sampling_params=sampling_params)
        outputs = [r.outputs[0].text.strip() for r in response]
        return outputs


class OpenAILLM:
    def __init__(self, model_name: str, base_url: str = None, api_key: str = None, num_concurrent: int = 1, **kwargs):
        if api_key is None:
            raise ValueError(
                "API key not found. Please set the `OPENAI_API_KEY` environment variable."
            )
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key, **kwargs)
        self.model_name = model_name
        self.num_concurrent = num_concurrent

    def chat(self, messages, temperature=0.0, max_tokens=512):
        if self.num_concurrent > 1:
            outputs = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_concurrent) as executor:
                futures = [
                    executor.submit(
                        self._chat,
                        message,
                        temperature,
                        max_tokens,
                    )
                    for message in messages
                ]
                for future in tqdm.tqdm(futures, total=len(messages)):
                    outputs.append(future.result())
        else:
            outputs = [
                self._chat(message, temperature, max_tokens)
                for message in tqdm.tqdm(messages)
            ]
        return outputs


    def _chat(self, message, temperature, max_tokens):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=message,
            max_completion_tokens=max_tokens,
            temperature=temperature,
        )
        output = response.choices[0].message.content.strip()
        return output


################fuck


class LocalAPILLM:
    """本地API模型实现"""
    
    def __init__(self, api_url: str, model_name: str, tokenizer_name: str, num_concurrent: int = 1, max_retries: int = 20, timeout: int = 600, **kwargs):
        self.api_url = api_url
        self.num_concurrent = num_concurrent
        self.max_retries = max_retries
        self.timeout = timeout
        self.generation_config = kwargs.get("generation_config", {})
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # 验证API端点
        # self._validate_api_endpoint()
        logger.info(f"Initialized LocalAPILLM with endpoint: {self.api_url}, model: {model_name}")
      
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _send_request(self, payload: dict):
        """发送API请求（带重试机制）"""
        try:
            start_time = time.time()
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            # 解析响应
            result = response.json()
            latency = time.time() - start_time
            logger.debug(f"API call succeeded in {latency:.2f}s")
            
            return result
        except requests.exceptions.RequestException as e:
            error_msg = f"API request failed: {str(e)}"
            if hasattr(e, 'response') and e.response is not None:
                error_msg += f", Response: {e.response.text[:200]}"
            logger.error(error_msg)
            raise
        except json.JSONDecodeError:
            logger.error("Invalid JSON response from API")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during API call: {str(e)}")
            raise
    
    def _prepare_payload(self, messages: list) -> dict:
        """准备API请求负载"""
        payload = {
            "prompt": messages,
            "max_tokens": self.generation_config.get("max_tokens", 512),
            "temperature": self.generation_config.get("temperature", 0.0),
        }
        
        # 添加可选参数
        for param in ["top_p", "top_k", "do_sample"]:
            if param in self.generation_config:
                payload[param] = self.generation_config[param]
        
        return payload
    
    def chat(self, messages_list: list, temperature=0.0, max_tokens=512) -> list:
        """处理聊天请求"""
        # 覆盖默认参数
        self.generation_config["temperature"] = temperature
        self.generation_config["max_tokens"] = max_tokens
        
        if self.num_concurrent > 1:
            return self._concurrent_chat(messages_list)
        else:
            return self._sequential_chat(messages_list)
    
    def _concurrent_chat(self, messages_list: list) -> list:
        """并发处理请求"""
        results = [None] * len(messages_list)
        messages_list = [self.tokenizer.apply_chat_template(prompt,tokenize=False,add_generation_prompt=True) for prompt in messages_list]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_concurrent) as executor:
            futures = {
                executor.submit(self._process_single, messages): idx
                for idx, messages in enumerate(messages_list)
            }
            
            for future in tqdm.tqdm(
                concurrent.futures.as_completed(futures),
                total=len(messages_list),
                desc="API Generating"
            ):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.error(f"Request failed for index {idx}: {str(e)}")
                    results[idx] = ""
        
        return results
    
    def _sequential_chat(self, messages_list: list) -> list:
        """顺序处理请求"""
        messages_list = [self.tokenizer.apply_chat_template(prompt,tokenize=False,add_generation_prompt=True) for prompt in messages_list]
        # import pdb;pdb.set_trace()
        results = []
        for messages in tqdm.tqdm(messages_list, desc="API Generating"):
            try:
                results.append(self._process_single(messages))
            except Exception as e:
                logger.error(f"Request failed: {str(e)}")
                results.append("")
        return results
    
    def _process_single(self, messages: list) -> str:
        """处理单个请求"""
        payload = self._prepare_payload(messages)
        response = self._send_request(payload)
        # import pdb;pdb.set_trace()
        
        # 解析响应
        if "choices" not in response or len(response["choices"]) == 0:
            raise ValueError("Invalid API response format: missing choices")
        
        return response["choices"][0]['text'].strip()

