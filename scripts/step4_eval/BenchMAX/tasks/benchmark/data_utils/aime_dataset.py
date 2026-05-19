from datasets import load_dataset
from .base_dataset import BaseDataset
from datasets import Dataset
import os
import csv

class AimeDataset(BaseDataset):
    def load_data(self) -> Dataset:
        return load_dataset("parquet", 
            data_files=["/mnt/nas/users/zhouhao.zh/latentcot/code/deepscaler/deepscaler/data/aime.parquet"],
            split=self.config.split
        )
        # ).select(range(1))
    def get_prompt_template(self) -> dict:
        return None
    
    def prepare_inputs(self, example, tokenizer) -> str:
        return tokenizer.apply_chat_template(example[self.config.prompt_key],tokenize=False,add_generation_prompt=True)
    
    #Arena has no references
    def get_references(self, example) -> list:
        return example["reward_model"]
    def save_results(self, predictions, references, metrics, output_dir):

        # Check if file exists
        csv_path = os.path.join(output_dir, 'pass.csv')
        file_exists = os.path.isfile(csv_path)
        # Write to CSV
        with open(csv_path, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=metrics.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(metrics)

        def add_responses_column(example, idx):
            example["responses"] = predictions[idx]
            return example

        updated_dataset = self.dataset.map(
            add_responses_column,
            with_indices=True,  # 传递索引以便访问responses列表
            remove_columns=None  # 保留所有原始列
        )
        updated_dataset.to_parquet(os.path.join(output_dir, "aime.parquet"))

        # import json
        # import os

        # filename = f"arena.jsonl"
        # os.makedirs(output_dir, exist_ok=True)
        
        # 创建arena评估所需的格式
        # question_ids = self.get_question_ids()
        # arena_format = []
        # for i, (pred, q_id) in enumerate(zip(predictions, question_ids)):
        #     arena_format.append({
        #         "uid": q_id,
        #         "prediction": pred,
        #         "model_id":output_dir.split("/")[-2], #.../model_name/lang
        #     })
                 

        # with open(os.path.join(output_dir, filename), "w") as f:
        #     for item in arena_format:
        #         f.write(json.dumps(item,ensure_ascii=False) + "\n") 
