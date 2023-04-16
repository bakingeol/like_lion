#%%
from transformers import AutoTokenizer
from transformers import AutoModelForMaskedLM
import torch
# model = AutoModelForMaskedLM.from_pretrained("snunlp/KR-BERT-char16424") #- prompt_list_dataofbirth 에서만 의미있는 결과가 나옴
# tokenizer = AutoTokenizer.from_pretrained("snunlp/KR-BERT-char16424") #- prompt_list_dataofbirth 에서만 의미있는 결과가 나옴
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = AutoModelForMaskedLM.from_pretrained("klue/roberta-large").to(device)
tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large")
# model = AutoModelForMaskedLM.from_pretrained("skt/kobert-base-v1")
# tokenizer = AutoTokenizer.from_pretrained("skt/kobert-base-v1")
# %%
from preprocess import RE, label_extract

#%% ######################################################################################## 전처리
def label_extract(label_extract,train):
    return_list = []
    for i in range(len(train)):
        if train[i]['label'] == f'{label_extract}':
            return_list.append(train[i])
    return return_list
#%%
file_path = '/home/lab614/ingeol/prompt_project/klue-re-v1.1/klue-re-v1.1_train.json'
data_load = RE.load(file_path)
# %%
train, dev = RE.split(data_load)
len(train)
#%%
# %%
index_list = ['per:date_of_birth','org:top_members/employees','per:employee_of','per:alternate_names','org:product']
data_of_birth, top_members, employee_of, alternate_names, product = [label_extract(i,train) for i in index_list]
#%%
len(data_of_birth), len(top_members), len(employee_of), len(alternate_names), len(product)
#%%
data_of_birth
#%%
from transformers import FillMaskPipeline
pip = FillMaskPipeline(model=model, tokenizer=tokenizer)
#%%
def generate_x_mask(subject):
    x = []
    for i in range(len(subject)):
        x.append((subject[i]['subject_entity']['word'],subject[i]['object_entity']['word']))
    return x    
data_of_birth_sub_ob, top_members_sub_ob, employee_of_sub_ob, alternate_names_sub_ob, product_sub_ob = map(generate_x_mask,
                                                                                                           [data_of_birth, top_members, employee_of, alternate_names, product])
#%%
label_list = []
for i in train:
    label_list.append(i['label'])
label_list = set(label_list)
label_list = list(label_list)
label_list
#%%
import re
#%%
relation_name = []
for i in range(len(label_list)):
    globals()[f'label_{i}'] = label_extract(label_list[i],train)
    a = label_list[i].replace(':','_').replace('/','_')
    globals()[f'{a}'] = globals()[f'label_{i}']
    relation_name.append(a)
#%%
obj_sub = []
obj_sub = list(map(generate_x_mask,[globals()[f'{relation_name[i]}'] for i in range(len(relation_name))]))
#%%
relation_name
#%%
obj_sub[0]
#%% ######################################################################################## 실험
org_dissolved_prompt = ['[x]는 [MASK]에 망했다.', '[x]는 [MASK]에 해산되었다.', '[x]는 [MASK]에 해산했다.'] 
# 0.0 0.0 0.0
per_alternate_names_prompt = ['[x]는 [MASK]라고도 불린다.', '[x]는 [MASK]라고 불린다.', '[x]는 [MASK]이다.'] 
# 0.0022522522522522522 0.0022522522522522522 0.0
org_political_religious_affiliation_prompt = ['[x]는 [MASK]이념을 가지고 있다.', '[x]는 [MASK] 사상이다.', '[x]는 [MASK]에 소속되어 있다.'] #프롬프트 보완 필요성 있음
# 0.10344827586206896 0.10344827586206896 0.022988505747126436
org_number_of_employees_members_prompt = ['[x]는 [MASK]에 속한다.', '[x]는 [MASK]에 속해 있다.', '[x]는 [MASK]에 소속되어 있다.'] 
# 0.0 0.0 0.0 숫자
per_employee_of_prompt = ['[x]는 [MASK]에 속한다.', '[x]는 [MASK]에 속해 있다.', '[x]는 [MASK]에 소속되어 있다.'] 
#0.0 0.011187072715972654 0.0052827843380981974
per_siblings_prompt = ['[x]는 [MASK]와 형제이다.', '[x]는 [MASK]와 가족이다.', '[x]는 [MASK]와 동생이다.'] 
# 0.0 0.0 0.0
org_founded_prompt = ['[x]는 [MASK]에 설립되었다.', '[x]는 [MASK]에 설립되었다.', '[x]는 [MASK]에 설립되었다.'] 
#0.0 0.0 0.0# 숫자 ~~년, ~~월, ~~일// ~~년
per_children_prompt = ['[x]는 [MASK]와 아들이다.', '[x]는 [MASK]와 딸이다.', '[x]는 [MASK]와 가족이다.'] 
# 0.0 0.0 0.0
per_colleagues_prompt = ['[x]는 [MASK]와 동료이다.', '[x]는 [MASK]의 지인이다.', '[x]는 [MASK]를 안다.'] 
# 0.0 0.06079664570230608 0.0
per_schools_attended_prompt = ['[x]는 [MASK]에 다녔다.', '[x]는 [MASK] 학교 출신다.', '[x]는 [MASK]에 다닌다.'] 
# 0.0 0.0 0.0
org_product_prompt = ['[x]는 [MASK]을 생산한다.', '[x]는 [MASK]을 만들었다.', '[x]는 [MASK]을 판매한다.'] 
# 0.008620689655172414 0.008620689655172414 0.011494252873563218 
per_other_family_prompt = ['[x]는 [MASK]와 가족이다.', '[x]는 [MASK]를 알고있다.', '[x]는 [MASK]와 형제이다.'] 
# 0.0 0.0 0.0
per_place_of_residence_prompt = ['[x]는 [MASK]에 산다.', '[x]는 [MASK]에 지낸다.', '[x]는 [MASK] 출신이다.'] 
# 0.0055248618784530384 0.0 0.09392265193370165
per_religion_prompt = ['[x]는 [MASK]에 종교를 가진다.', '[x]는 [MASK] 종교이다.', '[x]는 [MASK]를 믿는다.'] 
# 0.01098901098901099 0.0 0.0
org_top_members_employees_prompt = ['[x]는 [MASK]에 속한다.', '[x]는 [MASK]에 속해 있다.', '[x]는 [MASK]에 소속되어 있다.'] 
# 0.0 0.0 0.0 # 지역 이름을 받고 사람을 맞출 수 없음
per_date_of_birth_prompt = ['[x]는 [MASK]에 태어났다.', '[x]는 [MASK]에 태어났다.', '[x]는 [MASK]에 태어났다.']
# 0.0 0.0 0.0 # 숫자
per_origin_prompt = ['[x]는 [MASK]국적을 가진 구성원이다.', '[x]는 [MASK] 국적을 가진 국민이다.', '[x]는 [MASK] 국적을 가진다.'] #실험 가능 데이터
# 0.25040518638573744 0.2633711507293355 0.23612334801762114
org_place_of_headquarters_prompt = ['[x]는 [MASK]에 본사가 있다.', '[x]는 [MASK]에 위치한다.', '[x]는 [MASK]에 있다.'] # 실험 가능 데이터
# 0.18598130841121496 0.1411214953271028 0.0514018691588785 -> soft ensemble : 0.17850467289719626
org_founded_by_prompt = ['[x]는 [MASK]가 설립했다.', '[x]는 [MASK]이 만들었다.', '[x]는 [MASK]이 대표이다.'] 
# 0.0 0.007751937984496124 0.0
per_date_of_death_prompt = ['[x]는 [MASK]에 죽었다.', '[x]는 [MASK]에 죽었다.', '[x]는 [MASK]에 죽었다.'] 
# 0.0 0.0 0.0 # 숫자
per_place_of_birth_prompt = ['[x]는 [MASK]에서 태어났다.', '[x]는 [MASK]지냈다.', '[x]는 [MASK]에서 살았다.'] 
# 0.0945945945945946 0.0 0.006756756756756757
org_alternate_names_prompt = ['[x]는 [MASK]라고도 불린다.', '[x]는 [MASK]라고 불린다.', '[x]는 [MASK]라는 다른 명칭이 있다.'] # 실험 가능 데이터
# 0.18021793797150043 0.18189438390611903 0.18440905280804695
per_parents_prompt = ['[x]는 [MASK]와 아버지이다.', '[x]는 [MASK]와 어머니이다.', '[x]는 [MASK]와 가족이다.'] 
# 0.0 0.0 0.0
per_place_of_death_prompt = ['[x]는 [MASK]에서 죽었다.', '[x]는 [MASK]에서 사망했다.', '[x]는 [MASK]에서 생을 마감했다.'] 
# 0.0 0.0 0.0
org_member_of_prompt = ['[x]는 [MASK]에 속한다.', '[x]는 [MASK]에 속해 있다.', '[x]는 [MASK]에 포함된다.'] #프롬프트 보완 필요성 있음
# 0.07769869513641756 0.1073546856465006 0.05871886120996441
per_spouse_prompt = ['[x]는 [MASK]와 남편이다.', '[x]는 [MASK]와 아내이다.', '[x]는 [MASK]와 결혼했다.'] 
# 0.0 0.009790209790209791 0.009790209790209791
per_product_prompt = ['[x]는 [MASK]을 생산한다.', '[x]는 [MASK]을 만들었다.', '[x]는 [MASK]을 판매한다.'] 
# 0.008064516129032258 0.008064516129032258 0.0
org_members_prompt = ['[x]는 [MASK]를 포함한다.', '[x]는 [MASK]보다 크다.', '[x]는 [MASK]에 소속되어 있다.'] 
# 0.0 0.007832898172323759 0.0026109660574412533
per_title_prompt = ['[x]는 [MASK]이다.', '[x]는 [MASK]이다.', '[x]는 [MASK]이다.'] 
#0.0015789473684210526 0.0015789473684210526 0.0015789473684210526 # 이름을 보고 유추하기 어려움
no_relation_prompt = ['[x]는 [MASK]와 관련이 없다.', '[x]는 [MASK]와 관련이 없다.', '[x]는 [MASK]와 관련이 없다.'] 
# 0.004094046087261668 0.004094046087261668 0.004094046087261668# pass
#%%
len(per_origin), len(org_place_of_headquarters)
#%% relation_name, obj_sub

org_place_of_headquarters
# %% 3개 프롬프트에 대한 정확성 파악
def accuracy(prompt_list,subject_object):
    count,count2, count3 = 0,0,0
    for i in range(len(subject_object)):
        a,b,c = prompt_list[0].replace('[x]',subject_object[i][0]),prompt_list[1].replace('[x]',subject_object[i][0]),prompt_list[2].replace('[x]',subject_object[i][0])
        
        max_data_1 = max(pip(a), key=lambda x: x['score'])
        max_data_2 = max(pip(b), key=lambda x: x['score'])
        max_data_3 = max(pip(c), key=lambda x: x['score'])
        if max_data_1['token_str'] == subject_object[i][1]:
            count += 1
        if max_data_2['token_str'] == subject_object[i][1]:
            count2 += 1
        if max_data_3['token_str'] == subject_object[i][1]:
            count3 += 1
    return count/len(subject_object), count2/len(subject_object), count3/len(subject_object)
#%% 3개 프롬프트에 대한 실험
from tqdm import tqdm
for i in tqdm(range(len(relation_name))):
    a,b,c = accuracy(globals()[f'{relation_name[i]}_prompt'],obj_sub[i])
    print(f'{relation_name[i]} :',a,b,c)
#%% 프롬프트 한 개에 대한 실험
def accuracy_one(prompt_list,subject_object):
    count=0
    for i in range(len(subject_object)):
        a = prompt_list.replace('[x]',subject_object[i][0])
        max_data = max(pip(a), key=lambda x: x['score'])
        if max_data['token_str'] == subject_object[i][1]:
            count += 1
    return count/len(subject_object)
#%% 프롬프트 1개에 대한 실험 '나라 국민이다.'
prompt='[x]는 [MASK]에 본부가 있다.'

result = accuracy_one(prompt, obj_sub[23])
result
# %%
result
#%% ######################################################################################## 앙상블 - 소프트 보팅
import numpy as np

# 주어진 데이터
data = [
    {'score': 0.043343231081962585, 'token': 3671, 'token_str': '서울', 'sequence': '이순신는 서울 에서 태어났다.'},
    {'score': 0.045538727194070816, 'token': 1, 'token_str': '[PAD]', 'sequence': '이순신는 에서 살고 있다.'},
    {'score': 0.06821455806493759, 'token': 3671, 'token_str': '서울', 'sequence': '이순신는 서울 출생이다.'}
]

# 소프트 보팅으로 예측 확률값을 계산하여 가장 높은 값을 가진 클래스를 선택
class_probs = np.zeros(32000) # 클래스 개수는 32000
for d in data:
    class_probs[d['token']] += d['score']
y_soft_pred = np.argmax(class_probs)

print("Soft voting prediction:", y_soft_pred)
tokenizer.convert_ids_to_tokens(int(y_soft_pred))
#%%

def accuracy(prompt_list,subject_object):
    count,count2, count3 = 0,0,0
    for i in range(len(subject_object)):
        a,b,c = prompt_list[0].replace('[x]',subject_object[i][0]),prompt_list[1].replace('[x]',subject_object[i][0]),prompt_list[2].replace('[x]',subject_object[i][0])
        
        max_data_1 = max(pip(a), key=lambda x: x['score'])
        max_data_2 = max(pip(b), key=lambda x: x['score'])
        max_data_3 = max(pip(c), key=lambda x: x['score'])
        if max_data_1['token_str'] == subject_object[i][1]:
            count += 1
        if max_data_2['token_str'] == subject_object[i][1]:
            count2 += 1
        if max_data_3['token_str'] == subject_object[i][1]:
            count3 += 1
    return count/len(subject_object), count2/len(subject_object), count3/len(subject_object)
#%%
len(obj_sub)
#%%
#prompt_list = ['[x]는 [MASK]이념을 가지고 있다.', '[x]는 [MASK] 사상이다.', '[x]는 [MASK]에 소속되어 있다.']
import numpy as np
count = 0
def accuracy_soft_voting(prompt_list,obj_sub):
    max_data = []
    count=0
    for i in range(len(obj_sub)):
        a,b,c = prompt_list[0].replace('[x]',obj_sub[i][0]),prompt_list[1].replace('[x]',obj_sub[i][0]),prompt_list[2].replace('[x]',obj_sub[i][0])
        data = []
        for j in range(5):
            data.append(pip(a)[j])
            data.append(pip(b)[j])
            data.append(pip(c)[j])  
        
        class_probs = np.zeros(32001) # 클래스 개수는 3672 (0~3671)
        for d in data:
            class_probs[d['token']] += d['score']
        y_soft_pred = np.argmax(class_probs)
        
        answer_word = tokenizer.convert_ids_to_tokens(int(y_soft_pred))

        if answer_word == obj_sub[i][1]:
            count += 1
    return count/len(obj_sub)
#%% soft voting test

from tqdm import tqdm
for i in tqdm(range(len(relation_name))):
    result = accuracy_soft_voting(globals()[f'{relation_name[i]}_prompt'],obj_sub[i])
    print(f'{relation_name[i]} :',result)
#%%
relation_name[7]
#%%
accuracy_soft_voting(per_origin_prompt,obj_sub[7])
#%% ######################################################################################## 연습코드
prompt_list_dataofbirth = ["[x]는 [MASK] 에서 태어났다.", "[x]는 [MASK] 에서 살고 있다.","[x]는 [MASK] 출생이다."]
for i in prompt_list_dataofbirth:
    i=i.replace('[x]','이순신')
    print(pip(j))
# %%
prompt_top_members = ["[x]는 [MASK] 에서 일한다.", "[x]는 [MASK] 에서 일하고 있다.", "[x]는 [MASK] 소속이다."]
for j in prompt_top_members:
    j=j.replace('[x]','조용성')
    print(pip(j)[0])
# %%
prompt_employee_of = ['[x]는 [MASK]에 속한다.', '[x]는 [MASK]에 속해 있다.', '[x]는 [MASK]에 소속되어 있다.']
for j in prompt_employee_of:
    j=j.replace('[x]','이명박')
    print(pip(j))
#%%
prompt_alternate_names =['[x]는 [MASK]라고도 불린다.', '[x]는 [MASK]라고 불린다.', '[x]는 [MASK]이다.']
for j in prompt_alternate_names:
    j=j.replace('[x]','이명박')
    print(pip(j)[0])
# %%
prompt_product = ['[x]는 [MASK]을 생산한다.', '[x]는 [MASK]을 만들었다.', '[x]는 [MASK]을 판매한다.']
for j in prompt_product:
    j=j.replace('[x]','삼성전자')
    print(pip(j)[0])
#%%
a1,a2,a3=accuracy(prompt_list_dataofbirth,data_of_birth_sub_ob)
print(a1,a2,a3)
# %%
b1,b2,b3=accuracy(prompt_top_members,top_members_sub_ob)
print(b1,b2,b3)
# %%
c1,c2,c3=accuracy(prompt_employee_of,employee_of_sub_ob)
print(c1,c2,c3)   
# %%
d1,d2,d3=accuracy(prompt_alternate_names,alternate_names_sub_ob)
print(d1,d2,d3)
#%%
e1,e2,e3=accuracy(prompt_product,product_sub_ob)
print(e1,e2,e3)