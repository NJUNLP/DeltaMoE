from dataclasses import dataclass
from typing import Literal, Optional

@dataclass 
class DatasetConfig:
    dataset: str  # 数据集标识符 (如 squad, xsum)
    split: str = "test"
    prompt_key: str = "default"
    language: str =  "en"
    
    # 数据集特定参数
    max_samples: Optional[int] = None  # 限制样本数量
    shuffle: bool = False
    seed: int = 42
    
    # 预处理参数
    truncation: bool = True

    #sample hyper
    sample_ratio:  Optional[float] = None
