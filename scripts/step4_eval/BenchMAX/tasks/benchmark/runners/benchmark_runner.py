from data_utils import get_dataset
from models import get_model
from evaluators import get_evaluator
from configs import ModelConfig, DatasetConfig
from transformers import AutoTokenizer
import os

def run_benchmark(
    model_config: ModelConfig,
    dataset_config: DatasetConfig,
    output_dir: str
):
    """执行完整评测流程"""
    # 1. 加载数据集处理类
    dataset_handler = get_dataset(dataset_config)
    
    # 2. 加载模型并生成
    model = get_model(model_config)
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_path,padding_side="left", 
)
    
    # 3. 准备输入
    inputs = [
        dataset_handler.prepare_inputs(example,tokenizer)
        for example in dataset_handler.dataset
    ]
    references = [
        dataset_handler.get_references(example)
        for example in dataset_handler.dataset
    ]
    # import pdb;pdb.set_trace()
    predictions = model.generate(inputs)
    
    # 4. 评估结果
    #modify by config
    evaluator = get_evaluator(dataset_config.dataset)
    metrics = evaluator.evaluate(predictions, references)
    
    # 5. 保存结果
    os.makedirs(output_dir, exist_ok=True)
    dataset_handler.save_results(
        predictions, 
        references, 
        metrics, 
        output_dir
    )
    
    return metrics