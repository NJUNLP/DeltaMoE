from datasets import load_dataset

dataset = load_dataset("alexandrainst/m_mmlu",'hu')
import pdb; pdb.set_trace()


"""
(Pdb) dataset['test']
Dataset({
    features: ['instruction', 'option_a', 'option_b', 'option_c', 'option_d', 'answer', 'id'],
    num_rows: 13024
})
(Pdb) dataset['answer']
*** KeyError: 'answer'
(Pdb) dataset['test']['answer'][0]
'B'
(Pdb) type(dataset['test']['answer'][0])
<class 'str'>
(Pdb) type(dataset['test']['instruction'][0])
<class 'str'>
"""
