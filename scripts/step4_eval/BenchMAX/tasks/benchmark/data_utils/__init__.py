from .belebele_dataset import BelebeleDataset
from .okpai_mmlu_dataset import MmmluDataset
from .arena_dataset import ArenaDataset
from .aime_dataset import AimeDataset
from .mt_aime_dataset import MT_AimeDataset

DATASET_REGISTRY = {
    "belebele": BelebeleDataset,
    "m_mmlu": MmmluDataset,
    "arena": ArenaDataset,
    "aime": AimeDataset,
    "mt_aime":MT_AimeDataset,
    # 添加更多数据集...
}

def get_dataset(config):
    """根据配置获取数据集处理类"""
    if config.dataset not in DATASET_REGISTRY:
        raise ValueError(f"Unsupported dataset: {config.dataset}")
    return DATASET_REGISTRY[config.dataset](config)