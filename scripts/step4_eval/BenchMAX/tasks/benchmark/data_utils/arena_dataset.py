from datasets import load_dataset
from .base_dataset import BaseDataset
from datasets import Dataset


class ArenaDataset(BaseDataset):
    def load_data(self) -> Dataset:
        return load_dataset("LLaMAX/BenchMAX_Model-based", f"{self.config.language}",split=self.config.split)

    def get_prompt_template(self) -> dict:
        return None
    
    def prepare_inputs(self, example, tokenizer) -> str:
        mesasgae = [
            {"role": "user", "content": example["turns"][0]['content']},
        ]
        return tokenizer.apply_chat_template(mesasgae,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    
    #Arena has no references
    def get_references(self, example) -> list:
        return None
    def get_question_ids(self) -> list:
        return self.dataset["question_id"]
    def save_results(self, predictions, references, metrics, output_dir):

        import json
        import os

        filename = f"arena.jsonl"
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建arena评估所需的格式
        question_ids = self.get_question_ids()
        arena_format = []
        for i, (pred, q_id) in enumerate(zip(predictions, question_ids)):
            arena_format.append({
                "uid": q_id,
                "prediction": pred,
                "model_id":output_dir.split("/")[-2], #.../model_name/lang
            })
                 

        with open(os.path.join(output_dir, filename), "w") as f:
            for item in arena_format:
                f.write(json.dumps(item,ensure_ascii=False) + "\n") 
