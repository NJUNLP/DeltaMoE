from dataclasses import dataclass,field
from typing import Literal, Optional

@dataclass
class ModelConfig:

    #model_type
    model_type: Literal["huggingface", "vllm"]
    model_path: str
    dtype: Literal["bfloat16", "float32"] = "bfloat16"
    max_retries: int = 10

    #api
    api_url: Optional[str] = None # API端点URL
    timeout: int = 30             # 请求超时时间（秒）
    workers: int = 500            # 并发工作线程数
    
    # 生成参数
    max_new_tokens: int = 2048
    temperature: float = 0
    do_sample: bool = True
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    n_samples: Optional[int] = 1
    stop_token_ids: Optional[list[int]] = field(default_factory=lambda: [151645, 151643]) #for qwen
    
    # 硬件参数
    batch_size: int = 8
    device_map: str = "auto"
    use_flash_attn: bool = True
    tensor_parallel_size: int = 1

    # generation_config: {
    #     "max_new_tokens": self.max_new_tokens,
    #     "temperature": self.temperature,
    #     "do_sample": self.do_sample,
    #     "top_p": self.top_p,
    #     "top_k": self.top_k,
    # }

    # def get_generation_config(self) -> dict:
    #     """转换为生成参数字典"""
    #     return {
    #         "max_tokens": self.max_new_tokens,
    #         "temperature": self.temperature,
    #         "do_sample": self.do_sample,
    #         "top_p": self.top_p,
    #         "top_k": self.top_k
    #     }
    
