from abc import ABC, abstractmethod
from configs.model_config import ModelConfig

class BaseModel(ABC):
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
    
    @abstractmethod
    def generate(self, inputs: list[str]) -> list[str]:
        """批量生成文本"""
        pass