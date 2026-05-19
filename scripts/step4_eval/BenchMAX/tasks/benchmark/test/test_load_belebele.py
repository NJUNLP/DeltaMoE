from datasets import load_dataset
from datasets import load_dataset, DatasetDict, concatenate_datasets
import numpy as np


dataset = load_dataset("facebook/belebele","eng_Latn")['test']
"""
DatasetDict({
    test: Dataset({
        features: ['link', 'question_number', 'flores_passage', 'question', 'mc_answer1', 'mc_answer2', 'mc_answer3', 'mc_answer4', 'correct_answer_num', 'dialect', 'ds'],
        num_rows: 900
    })
})
"""
import pdb; pdb.set_trace()

