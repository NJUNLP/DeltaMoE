from datasets import load_dataset
import argparse
from utils.common_belebele import MULTILINGUAL_ANSWER_REGEXES, MULTILINGUAL_ANSWER_PATTERN_TEMPLATE,normalize_extracted_answer
import re


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_file_path", type=str, default='/mnt/nas/zhouh/dataset/json')

    args = parser.parse_args()
    print(args.dataset_file_path)
    dataset = load_dataset("json", data_files=[args.dataset_file_path])['train']
    
    results = []
    for i in range(len(dataset)):
        # for prediction,reference in zip(predictions,references):
        prediction = dataset[i]['predictions']['answers']
        reference = dataset[i]['metrics']['reference']
        for answer_regex in MULTILINGUAL_ANSWER_REGEXES:
            regex = MULTILINGUAL_ANSWER_PATTERN_TEMPLATE.format(answer_regex)
            match = re.search(regex, prediction)
            extracted_answer = None
            if match:
                extracted_answer = normalize_extracted_answer(match.group(1))
                break
        score = 1.0 if extracted_answer == reference else 0.0
        results.append(score)

    acc = sum(results) / len(results)
    print(f"{args.dataset_file_path} ACC: {acc}")