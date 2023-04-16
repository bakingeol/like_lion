#%%
import json
from typing import List, Tuple, Dict, Any
import random
import torch
class RE:
    def __init__(self, data, indices):
        self._data = data
        self._indices = indices
    
    @classmethod
    def load(cls, file_path: str):
        with open(file_path, 'r', encoding = 'utf-8') as fd:
            data = json.load(fd)
            
        indices = []
        for index, value in enumerate(data):
            indices.append((index, value))
    
        return cls(data, indices)

    @classmethod
    def split(cls, dataset, eval_ratio: float=0, seed = 42):
        indices = list(dataset._indices)
        random.seed(seed)
        random.shuffle(indices)
        train_indices = indices[int(len(indices)*eval_ratio):]
        eval_indices = indices[:int(len(indices)*eval_ratio)]
        
        return cls(dataset._data, train_indices), cls(dataset._data, eval_indices)
    
    def __getitem__(self, index : int) -> Dict[str, Any]:
        loc = self._indices
        sentence = loc[index][1]['sentence']
        subject_entity = loc[index][1]['subject_entity']
        object_entity = loc[index][1]['object_entity']
        label = loc[index][1]['label']
        
        return {
            'sentence':sentence, 
            'subject_entity':subject_entity, 
            'object_entity':object_entity, 
            'label':label
        }
    def __len__(self) -> int:
        return len(self._indices)
# %%
def label_extract(label_extract,train):
    return_list = []
    for i in range(len(train)):
        if train[i]['label'] == f'{label_extract}':
            return_list.append(train[i])
    return return_list
# %%
def predict_masked_token(prompt, model, tokenizer):
    tokens = tokenizer.tokenize(prompt)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = torch.tensor([input_ids])
    inputs = model(input_ids)
    proba =torch.softmax(inputs.logits, dim =-1)
    top_predictions= torch.topk(proba, 5, dim=-1).indices[0].tolist()
    return tokenizer.convert_ids_to_tokens(top_predictions[0])

# %%
