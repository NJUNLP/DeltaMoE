from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
from torch.utils.data import DataLoader
from typing import List
from tqdm import tqdm

torch_dtype = "bfloat16"
model_path = "/mnt/nas_intern/users/zhouhao.zh/new_project/hf_ckpt/delta_by_tian/lf/merged_ct_10B_from_base_seq_1024_lr_1e-5"
tokenizer = AutoTokenizer.from_pretrained(model_path)



model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch_dtype,
    attn_implementation="flash_attention_2" 
).cuda()

generation_config = GenerationConfig(
    eos_token_id=[151645,151643],  # 自动获取
    #eos_token_id=[151652],  # 自动获取
    max_new_tokens=1024,
    do_sample=False,
)

query = "Raymond and Samantha are cousins. Raymond was born 6 years before Samantha. Raymond had a son at the age of 23. If Samantha is now 31, how many years ago was Raymond's son born?"
query = "I have 10 liters of orange drink that are two-thirds water and I wish to add it to 15 liters of pineapple drink that is three-fifths water. But as I pour it, I spill one liter of the orange drink. How much water is in the remaining 24 liters?"
query = tokenizer.apply_chat_template([{"role": "user", "content": query}],tokenize=False,add_generation_prompt=True)
# print(query)
# import pdb;pdb.set_trace()
result = model.generate(
    tokenizer.encode(query, return_tensors="pt").cuda(),
    max_new_tokens=1024,
    generation_config=generation_config)
print(result.size())
print(result)
result_text = tokenizer.batch_decode(result)[0]
print(result_text)
# print(tokenizer.eos_token)