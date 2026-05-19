from .base import BaseEvaluator
from .belebele import BelebeleEvaluator
from .mmlu import MmmluEvaluator
from .arena import ArenaEvaluator
from .aime import AimeEvaluator


EVALUATOR_REGISTRY = {
    "belebele": BelebeleEvaluator,
    "m_mmlu": MmmluEvaluator,
    "arena": ArenaEvaluator,
    "aime": AimeEvaluator,
    "mt_aime":AimeEvaluator,
    # 添加更多数据集...
}

# def get_evaluator(dataset_name: str) -> 'BaseEvaluator':
#     """根据数据集名获取评估器实例"""
#     # 动态导入避免循环依赖
#     if dataset_name == "belebele":
#         return EVALUATOR_REGISTRY[dataset_name]()
#     else:
#         raise ValueError(f"No evaluator for dataset: {dataset_name}")

def get_evaluator(dataset_name: str) -> 'BaseEvaluator':
    """根据配置获取数据集处理类"""
    if dataset_name not in EVALUATOR_REGISTRY:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return EVALUATOR_REGISTRY[dataset_name]()

