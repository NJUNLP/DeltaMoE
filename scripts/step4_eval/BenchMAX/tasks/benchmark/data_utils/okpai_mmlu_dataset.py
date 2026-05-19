from datasets import load_dataset
from .base_dataset import BaseDataset
from utils.common_mmlu import MULTILINGUAL_ANSWER_REGEXES, MULTILINGUAL_ANSWER_PATTERN_TEMPLATE, normalize_extracted_answer, MMLU_QUERY_TEMPLATE
from utils.common_mmlu import MMLU_TEMPLATE_DICT
from datasets import Dataset
import random
from datasets import Dataset

# # 使用示例
# reduced_dataset = reduce_dataset_proportionally(dataset["test"], ratio=0.1, seed=42)
# print(f"原始大小: {len(dataset['test'])}")
# print(f"缩小后大小: {len(reduced_dataset)}")

class MmmluDataset(BaseDataset):

    def reduce_dataset_proportionally(self,dataset, ratio=0.1, seed=42):
        """
        按类别等比例缩小数据集，固定随机种子确保可复现
        
        Args:
            dataset: 原始数据集 (datasets.Dataset)
            ratio: 抽样比例 (默认0.1)
            seed: 随机种子 (默认42)
        
        Returns:
            Dataset: 缩小后的新数据集
        """
        # 设置随机种子确保可复现
        random.seed(seed)
        
        # 提取所有id并解析类别
        ids = dataset["id"]
        categories = [id_.split("/")[0] for id_ in ids]
        
        # 按类别分组：记录每个类别对应的行索引
        category_to_indices = {}
        for idx, cat in enumerate(categories):
            category_to_indices.setdefault(cat, []).append(idx)
        
        # 分层抽样：每个类别抽取 ratio 比例的样本
        selected_indices = []
        for cat, indices in category_to_indices.items():
            n_total = len(indices)
            n_select = max(1, round(n_total * ratio))  # 至少保留1个样本
            selected_indices.extend(random.sample(indices, n_select))
        
        # 创建新数据集
        return dataset.select(selected_indices)
    
    def load_data(self) -> Dataset:
        dataset = load_dataset("alexandrainst/m_mmlu", f"{self.config.language}",split=self.config.split)
        if self.config.sample_ratio:
            dataset = self.reduce_dataset_proportionally(dataset, ratio=self.config.sample_ratio, seed=self.config.seed)
        return dataset

        """
        Dataset({
            features: ['instruction', 'option_a', 'option_b', 'option_c', 'option_d', 'answer', 'id'],
            num_rows: 13024
        })
        """

    def get_prompt_template(self) -> dict:
        return {"default":MMLU_TEMPLATE_DICT}
    
    def prepare_inputs(self, example, tokenizer) -> str:
        template = self.prompt_template[self.config.prompt_key][self.config.language]
        # 'flores_passage', 'question', 'mc_answer1', 'mc_answer2', 'mc_answer3', 'mc_answer4',
        new_example = {
            "question":example["instruction"],
            "A":example["option_a"],
            "B":example["option_b"],
            "C":example["option_c"],
            "D":example["option_d"],
        }
        mesasgae = [
            {"role": "user", "content": template.format(**new_example)}
        ]
        #find out add_generation_prompt=True,
        return tokenizer.apply_chat_template(mesasgae,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    
    def get_references(self, example) -> list:
        return example["answer"]
        # return int(ord(example["answer"]) - ord("A"))
    
    def save_results(self, predictions, references, metrics, output_dir):

        import json
        import os
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mmlu_{timestamp}.json"
        os.makedirs(output_dir, exist_ok=True)


        accuracy =  sum([res["score"] for res in metrics] )/ len(metrics)
        print(f"Accuracy: {accuracy:.4f}")

        
        # 创建mmlu评估所需的格式
        mmlu_format = []
        for i, (pred, refs, metric) in enumerate(zip(predictions, references, metrics)):
            mmlu_format.append({
                "id": f"{timestamp}_{i}",
                "instruct_follow": metric["extrcated_answer"] == None,
                "answers": pred,
            })
        
        with open(os.path.join(output_dir, filename), "w") as f:
            json.dump({
                "metrics": metrics,
                "predictions": mmlu_format

            }, f, indent=2)

        statistic_result_filename = os.path.join(output_dir, "statistic.json")
        accuracy =  sum([res["score"] for res in metrics] )/ len(metrics)
        with open(statistic_result_filename, "w") as f:
            json.dump({
                "accuracy": accuracy,
            }, f, indent=2)

        print(f"Accuracy: {accuracy:.4f}")

        