# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/language-modeling/run_clm.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from itertools import chain
from typing import Any

from .processor_utils import DatasetProcessor

LANG_TO_RECOVER = {"old": False, "new": True}

@dataclass
class PretrainDatasetProcessor(DatasetProcessor):
    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        # build grouped texts with format `X1 X2 X3 ...` if packing is enabled
        eos_token = "<|end_of_text|>" if self.data_args.template == "llama3" else self.tokenizer.eos_token
        text_examples = [messages[0]["content"] + eos_token for messages in examples["_prompt"]]

        if not self.data_args.packing:
            if getattr(self.tokenizer, "add_bos_token", False):
                text_examples = [self.tokenizer.bos_token + example for example in text_examples]

            result = self.tokenizer(
                text_examples, add_special_tokens=False, truncation=True, max_length=self.data_args.cutoff_len
            )
        else:
            tokenized_examples = None
            if "langs" in examples and examples["langs"][0] != None: 
                #For CS data
                tokenized_examples = {"input_ids": [], "attention_mask": [],"langs": []}
                for i in range(len(text_examples)):
                    tokenized_sents = self.tokenizer(text_examples[i].split("##@@B"), add_special_tokens=False)
                    langs_mask = [[int(lang)] * len(input_ids) for lang,input_ids in zip(examples["langs"][i].split(),tokenized_sents["input_ids"])]
                    
                    tokenized_sents["langs_mask"] = langs_mask
                    tokenized_sents = {k: list(chain(*tokenized_sents[k])) for k in tokenized_sents.keys()}
                    
                    tokenized_examples["input_ids"].append(tokenized_sents["input_ids"])
                    tokenized_examples["attention_mask"].append(tokenized_sents["attention_mask"])
                    tokenized_examples["langs"].append(tokenized_sents["langs_mask"])
            elif "language" in examples and examples["language"][0] != None:
                # For mono data 
                tokenized_examples = self.tokenizer(text_examples, add_special_tokens=False)
                langs = examples["language"]
                langs_mask = [[LANG_TO_RECOVER[k]] * len(v) for k, v in zip(langs, tokenized_examples["input_ids"])]
                tokenized_examples['langs'] = langs_mask

            else:    
                tokenized_examples = self.tokenizer(text_examples, add_special_tokens=False)
            concatenated_examples = {k: list(chain(*tokenized_examples[k])) for k in tokenized_examples.keys()}
            total_length = len(concatenated_examples[list(concatenated_examples.keys())[0]])
            block_size = self.data_args.cutoff_len
            total_length = (total_length // block_size) * block_size
            result = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            if getattr(self.tokenizer, "add_bos_token", False):
                for i in range(len(result["input_ids"])):
                    result["input_ids"][i][0] = self.tokenizer.bos_token_id

        return result

    def print_data_example(self, example: dict[str, list[int]]) -> None:
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(self.tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
