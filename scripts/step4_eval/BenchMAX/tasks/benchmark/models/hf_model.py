from transformers import AutoTokenizer, AutoModelForCausalLM,GenerationConfig
import torch
from torch.utils.data import DataLoader
from typing import List
from .base_model import BaseModel
from tqdm import tqdm

class HFModel(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        # 初始化tokenizer（启用填充但禁用警告）
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_path,
            padding_side="left", 
            truncation_side="left",
            use_fast=True
        )
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token  # 使用eos_token作为pad_token
        
        # 加载模型
        torch_dtype = torch.bfloat16 if cfg.dtype == "bfloat16" else torch.float32
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     cfg.model_path,
        #     torch_dtype=torch_dtype,
        #     attn_implementation="flash_attention_2" if cfg.use_flash_attn else None
        # ).cuda()
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_path,
            torch_dtype=torch_dtype,
            attn_implementation="flash_attention_2" if cfg.use_flash_attn else None,
            device_map="auto",
        )
        self.model.eval()
        
    def _prepare_batches(self, texts: List[str]):
        """将文本转换为批处理的tokenized输入"""
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True
        )
        return encodings

    def generate(self, inputs: List[str]) -> List[str]:
        """高效批处理生成"""
        all_outputs = []
        
        # 创建数据加载器进行批处理
        dataloader = DataLoader(inputs, batch_size=self.cfg.batch_size)
        
        with torch.no_grad(), torch.cuda.amp.autocast():
            for batch_texts in tqdm(dataloader):
                # 1. 批处理tokenize
                encodings = self._prepare_batches(batch_texts)
                input_ids = encodings["input_ids"].to(self.model.device)
                attention_mask = encodings["attention_mask"].to(self.model.device)
                

                generation_config = {
                    "max_new_tokens": self.cfg.max_new_tokens,
                    "temperature": self.cfg.temperature,
                    "do_sample": self.cfg.do_sample,
                    "top_p": self.cfg.top_p,
                    "top_k": self.cfg.top_k,
                    'use_cache': True,
                }
                eos_config = GenerationConfig(eos_token_id=self.cfg.stop_token_ids)
                
                # 2. 批处理生成
                # import pdb; pdb.set_trace()
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=eos_config,
                    **generation_config
                )
                
                # 3. 解码时跳过填充token
                batch_outputs = [
                    self.tokenizer.decode(
                        output[len(input_ids[i]):],  # 只解码生成部分
                        skip_special_tokens=True
                    )
                    for i, output in enumerate(outputs)
                ]
                
                all_outputs.extend(batch_outputs)
        
        return all_outputs

    def __del__(self):
        """清理GPU内存"""
        if hasattr(self, 'model'):
            del self.model
            torch.cuda.empty_cache()