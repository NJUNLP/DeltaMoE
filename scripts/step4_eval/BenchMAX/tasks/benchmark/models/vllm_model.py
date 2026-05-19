from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM
from .base_model import BaseModel

class VLLMModel(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        dtype = "bfloat16" if cfg.dtype == "bf16" else "float32"
        self.llm = LLM(
            model=cfg.model_path,
            dtype=dtype,
            gpu_memory_utilization=0.9,
            tensor_parallel_size=cfg.tensor_parallel_size  
        )
        # import pdb;pdb.set_trace()
        self.sampling_params = SamplingParams(
            temperature = cfg.temperature,
            max_tokens = cfg.max_new_tokens,
            top_p = cfg.top_p,
            top_k = cfg.top_k,
            n = cfg.n_samples,
            stop_token_ids = cfg.stop_token_ids
        )
    def generate(self, inputs: list[str]) -> list[str]:
        outputs = self.llm.generate(inputs, self.sampling_params)
        # import pdb;pdb.set_trace()
        if self.cfg.n_samples == 1:
            return [output.outputs[0].text for output in outputs]
        else:
            return [[output.outputs[i].text for i in range(len(output.outputs))] for output in outputs]
