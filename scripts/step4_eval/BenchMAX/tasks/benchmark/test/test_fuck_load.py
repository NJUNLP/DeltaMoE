from datasets import load_dataset
from datasets import load_dataset, DatasetDict, concatenate_datasets
import numpy as np


dataset = load_dataset("li-lab/MMLU-ProX", "hu")['test']

def sample_by_category(dataset, fraction=0.1):
    
    seed = 42
    np.random.seed(seed)

    categories = set(dataset.unique('category'))

    subsets = []
    for category in categories:

        cat_dataset = dataset.filter(lambda x: x['category'] == category)
        
        num_samples = len(cat_dataset) // 10
        if num_samples == 0:
            print(f"Category '{category}' has less than 10 samples, skipping...")
            continue
        
        indices = np.random.permutation(len(cat_dataset))[:num_samples]
        
        subset = cat_dataset.select(indices)
        subsets.append(subset)

    sampled_dataset = concatenate_datasets(subsets)

    print(f"Sampled dataset size: {len(sampled_dataset)}")

dataset = load_dataset("li-lab/MMLU-ProX","hu")['test']
sample_by_category(dataset)

    
import pdb;pdb.set_trace()

