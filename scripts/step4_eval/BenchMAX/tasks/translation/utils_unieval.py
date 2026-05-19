#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import os
import re
import jsonlines
import logging
import numpy as np
from typing import List
from sacrebleu.metrics import BLEU
from sacrebleu.tokenizers.tokenizer_zh import TokenizerZh
from sacrebleu.tokenizers.tokenizer_13a import Tokenizer13a


# from uniform_eval.tasks.base import Task, MultilingualTask
# from uniform_eval.tasks.translation_scoring_utils.comet import comet_score
# from uniform_eval.tasks.utils import wait_async
# from uniform_eval.support.constants import JobType
# from uniform_eval.support.job_utils import get_reference


INVALID_VALUE = -999999999

flores_lang_map = {
    "ar": "Arabic",
    "ceb": "Cebuano",
    "de": "German",
    "es": "Spanish",
    "fa": "Persian",
    "fr": "French",
    "he": "Hebrew",
    "hi": "Hindi",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "km": "Khmer",
    "ko": "Korean",
    "lo": "Lao",
    "ms": "Malay",
    "my": "Burmese",
    "nl": "Dutch",
    "pt": "Portuguese",
    "ru": "Russian",
    "tgl": "Tagalog",
    "th": "Thai",
    "tr": "Turkish",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "en": "English",
    "zh": "Chinese",
}

flores_lang_map_rev = dict((v, k) for k, v in flores_lang_map.items())

flores_core_langs = {
    'ar',
    'de',
    'es',
    'fr',
    'id',
    'it',
    'ja',
    'ko',
    'pt',
    'ru',
    'th',
    'vi',
    'en',
    'zh'
}


scorer = None

def get_bleu_scorer():
    global scorer

    if scorer is None:
        scorer = BLEU(tokenize="13a", effective_order=True)
    return scorer


# Tokenizer from https://code.alibaba-inc.com/funan.whr/sacrebleu4alitranx/blob/master/sacrebleu4alitranx/sacrebleu4alitranx.py
def trim_multiple_space(tokes):
    return ''.join(tokes).strip().split()


def wmt_postprocess(text: str, lang: str) -> str:
    text = text.strip()
    texts = list(x.strip() for x in text.split('\n'))
    texts = list(x for x in texts if x != '')
    text = '\n'.join(texts)
    text = tokenize(text, lang)
    return text


def compute_maximum_bleu_value(gen: str, ref: str, lang: str):
    gens = list(x.strip() for x in gen.split('\n'))
    gens = list(x for x in gens if x != '')

    gens_tokens = list(wmt_postprocess(x, lang) for x in gens)
    if len(gens_tokens) > 10:
        print(f'The number of lines is {len(gens_tokens)}, which probably exceeds the limit of translation output. We only use the first 10 lines for calculating the best BLEU region.')
        gens_tokens = gens_tokens[:10]

    duplicated_gens_tokens = list()
    duplicated_gens_tokens_keys = set()

    for gens_token in gens_tokens:
        if gens_token not in duplicated_gens_tokens_keys:
            duplicated_gens_tokens_keys.add(gens_token)
            duplicated_gens_tokens.append(gens_token)

    gens_tokens = duplicated_gens_tokens
    ref_tokens = wmt_postprocess(ref, lang)

    scorer = get_bleu_scorer()

    maximum_bleu_value = -100.0
    maximum_bleu_object = None
    maximum_bleu_region = None

    for i in range(0, len(gens_tokens)):
        for j in range(i, len(gens_tokens)):
            gens_tokens_region = ' '.join(gens_tokens[i : j + 1])
            sentence_bleu = scorer.sentence_score(gens_tokens_region, [ref_tokens])

            if sentence_bleu.score > maximum_bleu_value:
                maximum_bleu_value = sentence_bleu.score
                maximum_bleu_object = sentence_bleu
                maximum_bleu_region = (i, j + 1)

    if maximum_bleu_object is None:
        sentence_bleu = scorer.sentence_score('', [ref_tokens])
        return sentence_bleu, (-1, -1), ['']
    else:
        return maximum_bleu_object, maximum_bleu_region, gens[maximum_bleu_region[0] : maximum_bleu_region[1]]


class SpaceTokenizer(object):
    def __call__(self, sent):
        if type(sent) == list:
            print(sent)
            raise ValueError()
        return " ".join(sent.strip().split())

class NonASCIITokenizer(object):
    def __init__(self):
        self.is_cjk = re.compile(
        "([\u2e80-\u9fff]|" # 中日韩
        "[\ua960-\ua97f]|"  # 谚文字母扩展A
        "[\uac00-\ud7ff]|" # 谚文音节+谚文字母扩展B
        "[\u0E00-\u0E7F]" # 泰文
        ")"
    )
    def __call__(self, sent):
        sent = sent.strip()
        chs = list(sent)
        line_chtok = []
        for ch in chs:
            if self.is_cjk.match(ch):
                line_chtok.append(' ')
                line_chtok.append(ch)
                line_chtok.append(' ')
            else:
                line_chtok.append(ch)
        line_chtok = trim_multiple_space(line_chtok)
        return ' '.join(line_chtok)

def build_tokenizer(lang: str):
    if lang == "Chinese":
        return TokenizerZh()
    elif lang in {"Japanese", "Korean", "Thai"}:
        return NonASCIITokenizer()
    else:
        return SpaceTokenizer()

def tokenize(sent, lang):
    tokenizer = build_tokenizer(lang)
    final_tokenizer = Tokenizer13a()
    return final_tokenizer(tokenizer(sent))


