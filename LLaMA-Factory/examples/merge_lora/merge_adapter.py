from peft import PeftModel  
from transformers import AutoModelForCausalLM,AutoTokenizer 
import argparse

args = argparse.ArgumentParser()
args.add_argument("--model_path", type=str, default="")
args.add_argument("--adapter_name_or_path", type=str,default="")
args.add_argument("--output_path", type=str,default="")

args = args.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model_path)
model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype="bfloat16")  
model = PeftModel.from_pretrained(model, args.adapter_name_or_path, torch_dtype="bfloat16")
merged_model = model.merge_and_unload()  
  
merged_model.save_pretrained(args.output_path)
tokenizer.save_pretrained(args.output_path)