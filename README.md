# klue prompt 프로젝트
- Model : "klue/roberta-large"
- Dataset : klue-re-v1.1_train  
- Method : manual, mining, paraphrasing
- Paper : https://arxiv.org/abs/1911.12543
- 결과 : RE 데이터 셋에서 [MASK] 단어를 추론하는데 대부분의 relation_entity 에서 낮은 결과가 나왔었고 그중 가장 점수가 높은 3개(per_origin_prompt, org_place_of_headquarters_prompt, org_alternate_names)에서 실험가능한 수치가 나왔다.
- Hard Voting 사용해 prompt3개, 5개의 정답률을 측정해본 결과 예상과 다르게 낮아지는 것이 확인되었다. 
- paraphrasing은 한국어와 어순이 같은 일본어로 진행했다.
- 메뉴얼은 실험자의 주관으로 만들었으며, 마이닝은 데이터의 문장에서 발취하였다.
- mining + manual(1) =5 에서 manual 데이터는 가장 정답률이 높은 데이터를 사용하였다.

<img width="1221" alt="main" src="file:///Users/baeg-ingeol/Documents/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202023-04-30%20%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB%2010.46.54.png">
