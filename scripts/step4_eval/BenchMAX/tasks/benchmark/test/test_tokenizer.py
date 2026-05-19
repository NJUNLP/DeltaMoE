from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("/mnt/nas/users/zhouhao.zh/LLMs/Qwen2.5-7B-Instruct")

# Prepare the input to the model
prompt = "Give me a short introduction to large language models."
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    # add_generation_prompt=True,
    enable_thinking=True,  # Set to False to strictly disable thinking
)

print(text)


"""
add_generation_prompt=True

<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
Give me a short introduction to large language models.<|im_end|>
<|im_start|>assistant
"""

"""
add_generation_prompt=False

<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
Give me a short introduction to large language models.<|im_end|>

"""

