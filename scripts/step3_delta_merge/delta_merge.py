import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import json


def main():
    parser = argparse.ArgumentParser(
        description="Delta merge: apply instruction-tuning delta to a CPT MoE model.\n"
                    "Formula: new_param = moe_param + instruct_param - base_param"
    )
    parser.add_argument("--base_model_path", type=str, required=True,
                        help="Path to the original dense base model (e.g. Qwen2.5-7B)")
    parser.add_argument("--cpt_model_path", type=str, required=True,
                        help="Path to the CPT-trained MoE checkpoint (output of Step 2)")
    parser.add_argument("--instruct_model_path", type=str, required=True,
                        help="Path to the instruction-tuned dense model (e.g. Qwen2.5-7B-Instruct)")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save the merged DeltaMoE model")
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.instruct_model_path)
    tokenizer.save_pretrained(args.output_path)

    config = json.load(open(os.path.join(args.cpt_model_path, "config.json")))
    config['model_type'] = "qwen2_moelpr"
    config["eos_token_id"] = 151645
    json.dump(config, open(os.path.join(args.output_path, "config.json"), 'w'), indent=2)

    print("Loading models...")
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model_path, torch_dtype=torch.bfloat16)
    instruct_model = AutoModelForCausalLM.from_pretrained(args.instruct_model_path, torch_dtype=torch.bfloat16)
    cpt_model = AutoModelForCausalLM.from_pretrained(args.cpt_model_path, torch_dtype=torch.bfloat16)

    base_state_dict = base_model.state_dict()
    instruct_state_dict = instruct_model.state_dict()
    cpt_state_dict = cpt_model.state_dict()

    config_data = json.load(open(os.path.join(args.cpt_model_path, "config.json")))
    num_experts = config_data['num_experts']

    new_state_dict = {}
    for key, value in cpt_state_dict.items():
        if 'mlp.experts' not in key:
            if "mlp.gate" in key:
                # Preserve router gate weights from CPT model
                new_state_dict[key] = cpt_state_dict[key]
            else:
                # Apply delta: moe + instruct - base
                new_state_dict[key] = cpt_state_dict[key] + instruct_state_dict[key] - base_state_dict[key]
            continue

        # MoE expert layers: broadcast the dense delta to each expert
        expert_idx = key.split('.')[5]
        base_key = key.replace(f'.experts.{expert_idx}', '')
        if base_key not in new_state_dict:
            new_state_dict[key] = cpt_state_dict[key] + instruct_state_dict[base_key] - base_state_dict[base_key]

    torch.save(new_state_dict, os.path.join(args.output_path, "pytorch_model.bin"))
    print(f"Merged DeltaMoE model saved to {args.output_path}")


if __name__ == "__main__":
    main()
