from .base import BaseEvaluator
from utils.common_mmlu import MULTILINGUAL_ANSWER_REGEXES, MULTILINGUAL_ANSWER_PATTERN_TEMPLATE,normalize_extracted_answer
import re

class MmmluEvaluator(BaseEvaluator):
    """计算精确匹配(EM)和F1分数"""
    
    def __init__(self):
        self.metric = "acc"
    
    def evaluate(self, predictions, references):

        results = []
        for prediction,reference in zip(predictions,references):
            for answer_regex in MULTILINGUAL_ANSWER_REGEXES:
                regex = MULTILINGUAL_ANSWER_PATTERN_TEMPLATE.format(answer_regex)
                match = re.search(regex, prediction)
                extracted_answer = None
                if match:
                    extracted_answer = normalize_extracted_answer(match.group(1))
                    break
            score = 1.0 if extracted_answer == reference else 0.0
            results.append(
                {
                    "score": score,
                    "extrcated_answer": extracted_answer,
                    "reference":  reference
                }
            )

        return results