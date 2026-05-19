from datasets import load_dataset
from datasets import load_dataset, DatasetDict, concatenate_datasets
import numpy as np


dataset = load_dataset("LLaMAX/BenchMAX_Model-based",'en')

import pdb; pdb.set_trace()

"""
(Pdb) dataset['train']['turns'][3]
[{'content': 'Design a semikinematic mounting for a right angle prism with preload provided by a compressed elastomeric pad. The mounting should be designed to ensure proper alignment of the prism with its mounting surface and provide adequate tension to maintain proper load transfer.'}]
(Pdb) dataset['train']]
*** SyntaxError: unmatched ']'
(Pdb) dataset['train']
Dataset({
    features: ['question_id', 'category', 'cluster', 'turns'],
    num_rows: 500
})
(Pdb) dataset['train']['turns'][3]
[{'content': 'Design a semikinematic mounting for a right angle prism with preload provided by a compressed elastomeric pad. The mounting should be designed to ensure proper alignment of the prism with its mounting surface and provide adequate tension to maintain proper load transfer.'}]
"""