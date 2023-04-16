#%%
from transformers import AutoModel, AutoTokenizer
from transformers import AutoModelForMaskedLM

model = AutoModelForMaskedLM.from_pretrained("snunlp/KR-BERT-char16424")
tokenizer = AutoTokenizer.from_pretrained("snunlp/KR-BERT-char16424")
# %%
from preprocess import RE, label_extract, predict_masked_token
import torch
#%%
file_path = '/home/lab614/ingeol/prompt_project/klue-re-v1.1/klue-re-v1.1_train.json'
data_load = RE.load(file_path)
# %%
train, dev = RE.split(data_load)
len(train)
# %%
index_list = ['per:date_of_birth','org:top_members/employees','per:employee_of','per:alternate_names','per:other_family','org:product']
data_of_birth, top_members, employee_of, alternate_names, other_family,product = [label_extract(i,train) for i in index_list]
#%%
len(data_of_birth), len(top_members), len(employee_of), len(alternate_names), len(other_family), len(product)
#%%
data_of_birth
#%%
from transformers import FillMaskPipeline
pip = FillMaskPipeline(model=model, tokenizer=tokenizer)
#%%
pip('나는 오늘 아침에 [MASK]에 출근을 했다.')
#%%
alternate_names
# %%
def predict_masked_token(prompt, subject,number,model, tokenizer):

    #prompt = '[x]는 [MASK]이다.'
    prompt = prompt.replace('[x]',subject[number]['subject_entity']['word'])
    x = len(tokenizer(f"{subject[number]['object_entity']['word']}")['input_ids'])-2
    prompt = prompt.replace('[MASK]','[MASK]'*(x))
    tokens = tokenizer.tokenize(prompt)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = torch.tensor([ids])
    inputs = model(input_ids)
    proba =torch.softmax(inputs.logits, dim =-1)
    top_predictions= torch.topk(proba, 5, dim=-1).indices[0].tolist()
    
    print('subject_entity : ',subject[number]['subject_entity']['word'],'object_entity : ', subject[number]['object_entity']['word'], '\n sentence : ',subject[number]['sentence'])
    for i in range(x):
        print(tokenizer.convert_ids_to_tokens(top_predictions[i]))
#%%
prompt = '[x]는 [MASK]라고 불린다.'
predict_masked_token(prompt, alternate_names,16,model, tokenizer)
#%%
#predict_masked_token(prompt, model, tokenizer)

#%%
def predict_masked_token(prompt, model, tokenizer): 
    tokens = tokenizer.tokenize(prompt)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = torch.tensor([input_ids])
    inputs = model(input_ids)
    proba =torch.softmax(inputs.logits, dim =-1)
    top_predictions= torch.topk(proba, 5, dim=-1).indices[0].tolist()
    return tokenizer.convert_ids_to_tokens(top_predictions[0])
#%%
data_of_birth[1]
# %%
prompt = '[x]의 생년월일은 [MASK] 입니다.'
prompt2 = '[x]의 생일은 [y] 입니다.'

# %%
x_input = data_of_birth[1]['subject_entity']['word']
#%%
prompt = prompt.replace('[x]',x_input)
x = tokenizer(data_of_birth[0]['sentence'], prompt, return_tensors='pt',padding='max_length',truncation=True)
x
# %%
tokens = tokenizer.tokenize(prompt)
input_ids = tokenizer.convert_tokens_to_ids(tokens)
input_ids = torch.tensor([input_ids])
tokens
# %%
a = tokenizer(prompt, return_tensors='pt',padding='max_length',truncation=True)
# %%
a
# %%
inputs = model(**a)
# %%
inputs.logits
#%%
proba =torch.softmax(inputs.logits, dim =-1)
# %%
top_predictions= torch.topk(proba, 5, dim=-1).indices[0].tolist()
# %%
top_predictions
# %%
tokenizer.convert_ids_to_tokens(top_predictions[0])
# %%

# %%
prompt2 = '[x]의 생일은 [MASK] 입니다.'
prompt2 = prompt2.replace('[x]',data_of_birth[0]['subject_entity']['word'])
prompt2 = prompt2.replace('[MASK]','[MASK]'*(len(tokenizer(f"{data_of_birth[0]['object_entity']['word']}")['input_ids'])-2))
prompt2
#%%
xz=predict_masked_token(prompt2,data_of_birth[0]['sentence'], model, tokenizer)
xz
#%%
data_of_birth[0]['sentence']
# %%
a = tokenizer(data_of_birth[0]['sentence'],prompt2)
a
# %%
b1=tokenizer.tokenize(data_of_birth[0]['sentence'], prompt2)
b2=tokenizer.convert_tokens_to_ids(b1)
b=torch.Tensor(a.input_ids)
c = torch.Tensor(b2)
model(c)
# %%
c=model(b)
c
# %%
