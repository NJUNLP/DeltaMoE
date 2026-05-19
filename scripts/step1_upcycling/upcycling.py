import argparse
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import torch
import os
import json


def main():
    parser = argparse.ArgumentParser(description="Upcycle a dense model to a Qwen2MoELPR MoE model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the original dense model")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the upcycled MoE model")
    parser.add_argument("--num_experts", default=4, type=int, help="Number of experts in the MoE model")
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.save_pretrained(args.output_path)

    config = json.load(open(os.path.join(args.model_path, "config.json"), "r"))
    config['architectures'] = ["Qwen2MoELPRForCausalLM"]
    config['num_experts'] = args.num_experts
    config['model_type'] = "qwen2_moelpr"
    json.dump(config, open(os.path.join(args.output_path, "config.json"), 'w'), indent=2)

    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    ckpt = model.state_dict()
    layers = len(model.model.layers)

    output = {}
    for i in range(layers):
        for k in ckpt:
            if ('layers.' + str(i) + '.') in k and 'mlp' not in k:
                output[k] = ckpt[k]
            else:
                for j in range(args.num_experts):
                    output[k.replace('mlp', 'mlp.experts.' + str(j))] = ckpt[k]

    torch.save(output, os.path.join(args.output_path, "pytorch_model.bin"))
    print(f"Upcycled model saved to {args.output_path}")


if __name__ == "__main__":
    main()
