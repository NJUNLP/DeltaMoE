from abc import ABC, abstractmethod

class BaseEvaluator(ABC):
    """所有评估器的基类"""  
    @abstractmethod
    def evaluate(self, predictions: list, references: list) -> dict:
        """返回评估指标字典"""
        pass

