from .base import BaseEvaluator

class SquadEvaluator(BaseEvaluator):
    """计算精确匹配(EM)和F1分数"""
    
    def __init__(self):
        # self.metric = load_metric("squad_v2")
        self.metric = "acc"    
    def evaluate(self, predictions, references):
        # 转换为SQuAD评估格式
        pass
        # formatted_preds = [
        #     {"id": str(i), "prediction_text": pred, "no_answer_probability": 0.0}
        #     for i, pred in enumerate(predictions)
        # ]
        
        # formatted_refs = [
        #     {"id": str(i), "answers": {"text": refs, "answer_start": [0]*len(refs)}}
        #     for i, refs in enumerate(references)
        # ]
        
        # results = self.metric.compute(
        #     predictions=formatted_preds,
        #     references=formatted_refs
        # )
        
        # return {
        #     "exact_match": results["exact"],
        #     "f1_score": results["f1"]
        # }