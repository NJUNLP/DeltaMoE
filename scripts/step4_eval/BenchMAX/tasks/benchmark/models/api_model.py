import time
import logging
import requests
import tenacity
from typing import List, Dict, Any
from .base_model import BaseModel
from configs.model_config import ModelConfig

logger = logging.getLogger(__name__)

class APIModel(BaseModel):
    """通用API模型实现，适配VLLM或其他兼容API的模型服务"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        # 验证API URL配置
        if not config.api_url:
            raise ValueError("api_url must be provided for APIModel")
        
        # 初始化配置
        self.api_url = config.api_url
        self.timeout = config.timeout
        self.workers = config.workers
        self.max_retries = config.max_retries # 固定重试次数
        
        # 日志配置
        logger.info(f"Initializing APIModel with endpoint: {self.api_url}")
        logger.info(f"Generation config: {config}")
        logger.info(f"Concurrency workers: {self.workers}")
    
    def _prepare_payload(self, prompt: str) -> Dict[str, Any]:
        """准备API请求负载"""
        # 获取生成配置
        
        # 构建基础负载
        payload = {
            "prompt": prompt,
            "n": 1
        }
        
        # 添加生成参数
        for param in ["max_tokens", "temperature", "do_sample", "top_p", "top_k"]:
            if hasattr(self.cfg, param):
                payload[param] = getattr(self.cfg, param)
        
        # 特殊处理：如果temperature为0，强制使用贪婪解码
        if "temperature" in payload and payload["temperature"] == 0:
            payload["do_sample"] = False
            if "top_p" in payload:
                del payload["top_p"]  # 贪婪解码时不需要top_p
            if "top_k" in payload:
                del payload["top_k"]  # 贪婪解码时不需要top_k
        
        return payload
    
    def _api_call(self, prompt: str) -> str:
        """执行单次API调用（不含重试）"""
        try:
            # 准备请求
            payload = self._prepare_payload(prompt)
            
            # 发送请求
            start_time = time.time()
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            # 解析响应
            result = response.json()
            
            # 检查响应格式
            if "choices" not in result or len(result["choices"]) == 0:
                raise ValueError("Invalid API response format: missing choices")
            
            generated_text = result['choices'][0]['text']
            
            # 记录性能
            latency = time.time() - start_time
            logger.debug(f"API call succeeded in {latency:.2f}s")
            
            return generated_text
        except requests.exceptions.RequestException as e:
            error_msg = f"API request failed: {str(e)}"
            if hasattr(e, 'response') and e.response is not None:
                error_msg += f", Response: {e.response.text[:200]}"
            logger.error(error_msg)
            raise
        except Exception as e:
            logger.error(f"Unexpected error during API call: {str(e)}")
            raise
    
    def _api_call_with_retry(self, prompt: str) -> str:
        """带多层重试机制的API调用"""
        for attempt in range(self.max_retries + 1):  # 0到max_retries次尝试
            try:
                return self._api_call(prompt)
            except Exception as e:
                if attempt < self.max_retries:
                    # 计算指数退避时间 (2^attempt 秒)
                    wait_time = min(2 ** attempt, 30)  # 最大等待30秒
                    logger.warning(
                        f"Attempt {attempt+1}/{self.max_retries} failed for prompt: {prompt[:50]}... "
                        f"Retrying in {wait_time}s. Error: {str(e)}"
                    )
                    time.sleep(wait_time)
                else:
                    # 最后一次尝试失败后抛出异常
                    logger.error(
                        f"All {self.max_retries} attempts failed for prompt: {prompt[:50]}... "
                        f"Final error: {str(e)}"
                    )
                    raise RuntimeError(f"API call failed after {self.max_retries} attempts") from e
    
    def generate(self, inputs: List[str]) -> List[str]:
        """批量生成文本（带自动重试机制）"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from tqdm import tqdm
        
        results = [None] * len(inputs)
        
        # 使用线程池处理并发请求
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            # 提交所有任务
            future_to_index = {
                executor.submit(self._api_call_with_retry, prompt): i
                for i, prompt in enumerate(inputs)
            }
            
            # 使用tqdm显示进度条
            with tqdm(total=len(inputs), desc="API Generating") as pbar:
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        results[index] = future.result()
                    except Exception as e:
                        logger.error(f"Final attempt failed for input {index}: {str(e)}")
                        results[index] = "ERROR"
                    finally:
                        pbar.update(1)
        
        return results