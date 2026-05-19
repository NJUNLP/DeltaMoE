from .base import BaseEvaluator
from utils.common_belebele import MULTILINGUAL_ANSWER_REGEXES, MULTILINGUAL_ANSWER_PATTERN_TEMPLATE,normalize_extracted_answer
import re
from openai import OpenAI


class ArenaEvaluator(BaseEvaluator):
    """计算精确匹配(EM)和F1分数"""
    
    def __init__(self):
        self.metric = "acc"
    
    #sk-1014b50fae1d4cf580c5d9382dc220d3
    def evaluate(self, predictions, references):
        return predictions

# Please install OpenAI SDK first: `pip3 install openai`


# client = OpenAI(api_key="sk-1014b50fae1d4cf580c5d9382dc220d3", base_url="https://api.deepseek.com")

# response = client.chat.completions.create(
#     model="deepseek-chat",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant"},
#         {"role": "user", "content": "Hello"},
#     ],
#     stream=False
# )

# print(response.choices[0].message.content)