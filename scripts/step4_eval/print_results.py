import os
import argparse
import glob
import json


def has_result_json_recursive(directory):
    matching_files = (
        glob.glob(os.path.join(directory, '**', 'result*.json'), recursive=True) +
        glob.glob(os.path.join(directory, '**', 'statistic.json'), recursive=True)
    )
    return matching_files[0] if matching_files else None


def xgsm_process(result_path, lang):
    path = has_result_json_recursive(result_path)
    if path:
        with open(path, "r", encoding='utf-8') as f:
            data = json.load(f)
            score = data["results"][f"xmgsm_native_cot_{lang}"]["exact_match,flexible-extract"] * 100
            return {f"xgsm_{lang}": str(score)}
    return {f"xgsm_{lang}": ""}


def xifeval_process(result_path, lang):
    path = has_result_json_recursive(result_path)
    if path:
        with open(path, "r", encoding='utf-8') as f:
            data = json.load(f)
            metrics = [
                float(data["results"][f"xifeval_{lang}"]["prompt_level_strict_acc,none"]),
                float(data["results"][f"xifeval_{lang}"]["inst_level_strict_acc,none"]),
                float(data["results"][f"xifeval_{lang}"]["prompt_level_loose_acc,none"]),
                float(data["results"][f"xifeval_{lang}"]["inst_level_loose_acc,none"]),
            ]
            score = sum(metrics) / len(metrics) * 100
            return {f"xifeval_{lang}": str(score)}
    return {f"xifeval_{lang}": ""}


def belebele_process(result_path, lang):
    path = has_result_json_recursive(result_path)
    if path:
        data = json.load(open(path))
        return {f"belebele_{lang}": str(data["accuracy"] * 100)}
    return {f"belebele_{lang}": ""}


def mmlu_process(result_path, lang):
    path = has_result_json_recursive(result_path)
    if path:
        data = json.load(open(path))
        return {f"mmlu_{lang}": str(data["accuracy"] * 100)}
    return {f"mmlu_{lang}": ""}


def flores_process(result_path, lang):
    flores_dict = {"flores_langtgt": "0", "flores_langsrc": "0"}

    if lang == "en":
        src_pattern = f'result_{lang}-zh.json'
        tgt_pattern = f'result_zh-{lang}.json'
    else:
        src_pattern = f'result_{lang}-en.json'
        tgt_pattern = f'result_en-{lang}.json'

    result_dir = "/".join(result_path.split('/')[:-1])
    src_files = glob.glob(os.path.join(result_dir, '**', src_pattern), recursive=True)
    tgt_files = glob.glob(os.path.join(result_dir, '**', tgt_pattern), recursive=True)

    if src_files:
        try:
            with open(src_files[0], 'r', encoding='utf-8') as f:
                data = json.load(f)
                flores_dict["flores_langsrc"] = str(data.get("xComet", 0) * 100)
        except (IOError, json.JSONDecodeError, KeyError) as e:
            print(f"Error processing source file {src_files[0]}: {e}")

    if tgt_files:
        try:
            with open(tgt_files[0], 'r', encoding='utf-8') as f:
                data = json.load(f)
                flores_dict["flores_langtgt"] = str(data.get("xComet", 0) * 100)
        except (IOError, json.JSONDecodeError, KeyError) as e:
            print(f"Error processing target file {tgt_files[0]}: {e}")

    return flores_dict


TASK2FUNC = {
    "xgsm": xgsm_process,
    "xifeval": xifeval_process,
    "belebele": belebele_process,
    "mmlu": mmlu_process,
    "flores": flores_process,
}

PRINT_MAPPING = {0: "xgsm", 1: "xifeval", 2: "belebele", 3: "mmlu", 4: "flores_tgt", 5: "flores_src"}


def get_res_dict(result_path, lang):
    return_dict = {}
    return_str = ""
    for task_name, process_func in TASK2FUNC.items():
        res_dict = process_func(os.path.join(result_path, task_name, lang), lang)
        return_dict[task_name] = res_dict
        return_str += ",".join(res_dict.values()) + ","
    return return_str, return_dict


def print_results(result_path, langs, label):
    print_dict = [""] * 6
    for lang in langs:
        res_str, _ = get_res_dict(result_path, lang)
        res_ele = res_str.split(",")[:-1]
        for i, val in enumerate(res_ele):
            if lang == "en" and i in (4, 5):
                continue
            print_dict[i] += val + ","

    print(f"\n=== {label} ===")
    for i, values in enumerate(print_dict):
        print(f"  {PRINT_MAPPING[i]}: {values}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str, required=True,
                        help="Path to the model's result directory (e.g. /results/Qwen2.5-7B-Instruct)")
    parser.add_argument("--expand_lang", nargs='+', default=["hu", "bn", "sr"],
                        help="Expanded (new) languages")
    parser.add_argument("--original_lang", nargs='+', default=["en", "es", "zh", "fr"],
                        help="Original languages")
    args = parser.parse_args()

    print_results(args.result_path, args.expand_lang, "Expanded Languages")
    print("-" * 40)
    print_results(args.result_path, args.original_lang, "Original Languages")


if __name__ == '__main__':
    main()
