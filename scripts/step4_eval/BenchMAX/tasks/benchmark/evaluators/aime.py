from .base import BaseEvaluator
import re
import numpy as np
from utils.r1 import extract_answer, grade_answer_sympy, grade_answer_mathd
import os

def reward_fn(response: str,ground_truth: str) -> bool:

    # Extract model answer.
    # import pdb;pdb.set_trace()
    model_answer = extract_answer(response)
    if model_answer is None:
        # print("false answer")
        # import pdb;pdb.set_trace()
        return False

    # Extract ground truth.
    ground_truth = str(ground_truth)
    if "\\boxed" in ground_truth:
        ground_truth = extract_answer(ground_truth)

    # print(f"model_answer:{model_answer}\n ground_truth:{ground_truth}")
    # Check against all possible correct answers
    is_correct = grade_answer_mathd(model_answer, ground_truth) or grade_answer_sympy(model_answer, ground_truth)
    return is_correct


class AimeEvaluator(BaseEvaluator):
    """计算精确匹配(EM)和F1分数"""
    
    def __init__(self):
        self.metric = "acc"    
    def evaluate(self, predictions, references):

        
        total_scores = []
        passes = 0
        for response_lst,reference in zip(predictions,references):
            score_lst = []
            for r in response_lst:
                # import pdb;pdb.set_trace()
                score = reward_fn(r,reference['ground_truth'])
                score_lst.append(score)
            max_score = np.max(score_lst)
            total_scores.append(score_lst)
            if max_score == 1:
                passes += 1
        
        pass_at_n = passes / len(predictions)
        pass_at_1 = np.mean(total_scores)

        n_samples = len(total_scores[0])
        
        print(f"pass@{n_samples}: {pass_at_n:.2%}")
        print(f"pass@1: {pass_at_1:.2%}")
        
        return {
            'pass@1': pass_at_1,
            f'pass@{n_samples}': pass_at_n,
        }


        # for response_lst,reference in zip(predictions,references):
        #     for answer_regex in MULTILINGUAL_ANSWER_REGEXES:
        #         regex = MULTILINGUAL_ANSWER_PATTERN_TEMPLATE.format(answer_regex)
        #         match = re.search(regex, prediction)
        #         extracted_answer = None
        #         if match:
        #             extracted_answer = normalize_extracted_answer(match.group(1))
        #             break
        #     score = 1.0 if extracted_answer == reference else 0.0
        #     results.append(
        #         {
        #             "score": score,
        #             "extrcated_answer": extracted_answer,
        #             "reference":  reference
        #         }
        #     )

        return results