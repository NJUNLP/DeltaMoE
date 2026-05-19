from typing import Union
from .base_model import BaseModel
from .hf_model import HFModel
from .vllm_model import VLLMModel
from .api_model import APIModel  # 新增导入
from configs.model_config import ModelConfig

def get_model(config: Union[ModelConfig, dict]) -> BaseModel:
    # 如果传入的是字典，转换为ModelConfig对象
    if isinstance(config, dict):
        config = ModelConfig(**config)
    
    # 类型检查确保配置合法
    if not isinstance(config, ModelConfig):
        raise TypeError(f"Expected ModelConfig, got {type(config)}")
    
    model_registry = {
        "hf": HFModel,
        "vllm": VLLMModel,
        "api": APIModel  # 新增API模型类型
    }
    
    try:
        model_class = model_registry[config.model_type]
    except KeyError:
        raise ValueError(
            f"Unsupported model_type: {config.model_type}. "
            f"Available options: {list(model_registry.keys())}"
        )
    
    try:
        # 特殊处理：API模型需要api_url
        if config.model_type == "api" and not config.api_url:
            raise ValueError("api_url is required for API model type")
        
        # 打印调试信息
        print(f"Initializing {config.model_type} model")
        if config.model_type == "api":
            print(f"API endpoint: {config.api_url}")
        else:
            print(f"Model path: {config.model_path}")
        print(f"Generation config: {config}")
        
        return model_class(config)
    except ImportError as e:
        raise ImportError(
            f"Failed to initialize {config.model_type} model. "
            f"Required dependencies: {_get_required_deps(config.model_type)}"
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"Model initialization failed with config: {config}"
        ) from e

def _get_required_deps(model_type: str) -> str:
    """返回不同模型需要的依赖说明"""
    deps = {
        "huggingface": "transformers>=4.40.0, torch>=2.2.0",
        "vllm": "vllm>=0.4.0, transformers",
        "api": "requests, tenacity"
    }
    return deps.get(model_type, "Unknown dependencies")

__all__ = ['get_model', 'BaseModel', 'HFModel', 'VLLMModel', 'APIModel']