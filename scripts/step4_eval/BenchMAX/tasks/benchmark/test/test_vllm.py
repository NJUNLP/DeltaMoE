from vllm import LLM, SamplingParams

def main():  
    # Configurae the sampling parameters (for thinking mode)
    sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=32768,n=3)

    # Initialize the vLLM engine
    llm = LLM(model="/mnt/nas/users/zhouhao.zh/LLMs/Qwen3/Qwen3-4B")

    # Prepare the input to the model
    prompt = "Give me a short introduction to large language models."
    messages = [
        {"role": "user", "content": prompt}
    ]

    # Generate outputs
    outputs = llm.chat(
        [messages] * 2, 
        sampling_params,
        # chat_template_kwargs={"enable_thinking": True},  # Set to False to strictly disable thinking
    )

    import pdb;pdb.set_trace()

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

if __name__ == "__main__":
    main()