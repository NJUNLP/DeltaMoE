from abc import ABC, abstractmethod
from datasets import Dataset

class BaseDataset(ABC):
    def __init__(self, config):
        self.config = config
        self.dataset = self.load_data()
        self.prompt_template = self.get_prompt_template()
    
    @abstractmethod
    def load_data(self) -> Dataset:
        """加载数据集并返回Dataset对象"""
        pass
    
    @abstractmethod
    def get_prompt_template(self) -> dict:
        """返回该数据集支持的提示词模板"""
        return {
            "default": "Default prompt template for {dataset_name}",
            "detailed": "Detailed prompt template for {dataset_name}"
        }
    
    @abstractmethod
    def prepare_inputs(self, example) -> str:
        """根据提示词模板准备模型输入"""
        template = self.prompt_templates[self.config.prompt_key]
        return template.format(**example)
    
    @abstractmethod
    def get_references(self, example) -> list:
        """获取参考答案（可能多个）"""
        return [example["answer"]]
    
    @abstractmethod
    def save_results(self, predictions, references, metrics, output_dir):
        """数据集特定的结果保存方法"""
        # 默认实现
        import json
        import os
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.config.dataset}_{self.config.model_path.split('/')[-1]}_{timestamp}.json"
        os.makedirs(output_dir, exist_ok=True)
        
        with open(os.path.join(output_dir, filename), "w") as f:
            json.dump({
                "config": self.config.__dict__,
                "metrics": metrics,
                "predictions": predictions,
                "references": references
            }, f, indent=2)