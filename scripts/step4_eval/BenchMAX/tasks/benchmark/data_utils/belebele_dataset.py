from datasets import load_dataset
from .base_dataset import BaseDataset
from utils.common_belebele import MULTILINGUAL_ANSWER_REGEXES, MULTILINGUAL_ANSWER_PATTERN_TEMPLATE, normalize_extracted_answer, BELEBELE_QUERY_TEMPLATE
from utils.common_belebele import BELEBELE_TEMPLATE_DICT
from datasets import Dataset

langMap = {}
langMap["el"]="ell_Grek"
langMap["tr"]="tur_Latn"
langMap["hu"]="hun_Latn"
langMap["en"]="eng_Latn"
langMap["es"]="spa_Latn"
langMap["ar"]="arb_Arab"
langMap["ru"]="rus_Cyrl"
langMap["de"]="deu_Latn"
langMap["zh"]="zho_Hans"
langMap["fr"]="fra_Latn"
langMap["pt"]="por_Latn"
langMap["ja"]="jpn_Jpan"
langMap["bn"]="ben_Beng"
langMap["sr"]="srp_Cyrl"
langMap["cs"]="ces_Latn"
langMap["ar"]="arb_Arab"
langMap["vi"]="vie_Latn"
langMap["te"]="tel_Telu"
langMap["cs"]="ces_Latn"
langMap["vi"]="vie_Latn"
langMap["is"]="isl_Latn"
langMap["tr"]="tur_Latn"
langMap["el"]="ell_Grek"

class BelebeleDataset(BaseDataset):
    def load_data(self) -> Dataset:
        return load_dataset("facebook/belebele", f"{langMap[self.config.language]}",split="test")

    def get_prompt_template(self) -> dict:
        return {"default":BELEBELE_TEMPLATE_DICT}
    
    def prepare_inputs(self, example, tokenizer) -> str:
        template = self.prompt_template[self.config.prompt_key][self.config.language]
        # 'flores_passage', 'question', 'mc_answer1', 'mc_answer2', 'mc_answer3', 'mc_answer4',
        new_example = {
            "passage": example["flores_passage"],
            "question":example["question"],
            "A":example["mc_answer1"],
            "B":example["mc_answer2"],
            "C":example["mc_answer3"],
            "D":example["mc_answer4"],
        }
        mesasgae = [
            {"role": "user", "content": template.format(**new_example)}
        ]
        #find out add_generation_prompt=True,
        return tokenizer.apply_chat_template(mesasgae,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    
    def get_references(self, example) -> list:
        return int(example["correct_answer_num"])
    
    def save_results(self, predictions, references, metrics, output_dir):

        import json
        import os
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"belebele_{timestamp}.json"
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建belebele评估所需的格式
        belebele_format = []
        for i, (pred, refs, metric) in enumerate(zip(predictions, references, metrics)):
            belebele_format.append({
                "id": f"{timestamp}_{i}",
                "instruct_follow": metric["extrcated_answer"] == None,
                "answers": pred,
            })
        
        with open(os.path.join(output_dir, filename), "w") as f:
            json.dump({
                "metrics": metrics,
                "predictions": belebele_format

            }, f, indent=2)

        statistic_result_filename = os.path.join(output_dir, "statistic.json")
        accuracy =  sum([res["score"] for res in metrics] )/ len(metrics)
        with open(statistic_result_filename, "w") as f:
            json.dump({
                "accuracy": accuracy,
            }, f, indent=2)

        print(f"Accuracy: {accuracy:.4f}")

        