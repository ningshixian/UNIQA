import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from zhon.hanzi import punctuation
import string
import re
import sys
sys.path.append(r"../")
from uniqa.components.preprocessors import cleaning_func

"""
【q-q相似句对数据】处理脚本
- 修改 train_data_path
- 删除【方案一】生成的相似句对中的无用base数据
- 数据清洗 & 删除无效数据
- 输出文件：processed_crm_similar_pair.csv

2024-09-10
知识平台相似问:  (26840, 2)
llm train_data:  (72931, 2)
aaq train_data_2:  (2602, 2)
所有q-q相似句对数据量： (101843, 2)

2024-09-11
知识平台相似问:  (26841, 2)
llm train_data:  (81520, 2)
llm train_data:  (62298, 2)
aaq train_data_2:  (2602, 2)
q-q相似句对数据量： (90177, 2)
"""

nb_lines = 10
train_data_1_path = '../data_factory/crm_similar_pair_0911_4training.csv'    # 【方案一】生成的相似句对数据✔
train_data_2_path = '../data_factory/crm_similar_pair_haixiao_2.csv'    # 【方案二】生成的相似句对数据✔
train_data_3_path = '../data_factory/qa4training.csv'  # 执行 check_redundancy.py 获取符合条件的相似问数据
proc_know_path = "../data_factory/processed_crm_similar_pair.csv"  # 写入文件路径

# nb_lines = 10
# train_data_1_path = 'crm_similar_pair_0911.csv'    # 【方案一】生成的相似句对数据✔
# train_data_2_path = 'crm_similar_pair_haixiao_2.csv'    # 【方案二】生成的相似句对数据✔
# train_data_3_path = 'qa4api.csv'  # 执行 check_redundancy.py 获取符合条件的相似问数据
# proc_know_path = "processed_crm_similar_pair.csv"  # 写入文件路径

# 拿到question 对应 base
know_list = pd.read_csv("crm_knowledge_common.csv", encoding="utf-8", index_col=False, keep_default_na=False)
ques2base = {know_list.loc[i,'question_content']:know_list.loc[i,'base_name'] for i in range(len(know_list))}

# 新增【知识平台】的相似问数据
# 执行 python check_redundancy.py 获取
kid2ques = {}
df_qa = pd.read_csv(train_data_3_path, encoding="utf-8", index_col=False, keep_default_na=False)
df_qa['question_id'] = df_qa['question_id'].astype(str)
for i in range(len(df_qa)):
    # question_id,question_content
    question_id = df_qa.loc[i,'question_id'].strip()
    question_content = df_qa.loc[i,'question_content'].strip()
    # answer_content = df_qa.loc[i,'answer_content'].strip()
    if question_id not in kid2ques:
        kid2ques[question_id] = [question_content]
    else:
        kid2ques[question_id].append(question_content)
train_data_qa = {"question_content":[],"question_generate":[]}
for qid, ques in kid2ques.items():
    if len(ques)>1:
        pri = ques[0]
        for sim in ques[1:]:
            train_data_qa["question_content"].append(pri)
            train_data_qa["question_generate"].append(sim)
train_data_qa = pd.DataFrame(train_data_qa)
print("知识平台相似问: ", train_data_qa.shape)  #  (28080, 2)

# 读取相似句对一
train_data = pd.read_csv(
    train_data_1_path,
    encoding="utf-8",
    index_col=False,
    dtype=str,  # 数据类型->字符串
)
train_data['question_content'] = train_data['question_content'].astype(str)
train_data['question_generate'] = train_data['question_generate'].astype(str)
print("llm train_data: ", train_data.shape)  #   (101430, 2)
# 添加question_base, 删除无用base数据
train_data['question_base'] = train_data['question_content'].apply(lambda x: ques2base[x] if x in ques2base else x)
base_no_use = ["热门知识库", "服务专家内部资料库", "内部测试库", "多媒体库"]
train_data = train_data[~train_data['question_base'].isin(base_no_use)]
# 删除无效数据
train_data = train_data[~(
    # (train_data["question_content"].str.match(r".*[\(（].*[\)）]$", na=False))  # 以括号结尾的行删除
    # (train_data["question_content"].str.match(r".*[\(（][a-zA-Z0-9]+[\)）]$", na=False))  # 以括号结尾，括号内仅数字或字母的行删除
    (train_data["question_content"].str.match(r".*[\(（]MEGA[\)）]$", na=False))  # 以括号结尾，括号内mega的行删除
    | (train_data["question_generate"].str.match(r".*[0-9]$", na=False))        # 相似问以数字结尾的行删除
    | (train_data["question_generate"].str.match(r".*：$", na=False))        # 相似问以冒号结尾的行删除
    | (train_data["question_generate"].str.len() <= 3)
)]
train_data.drop(columns=['question_base'], inplace=True)
print("llm train_data: ", train_data.shape)  #  (72931, 2)

# 读取相似句对二
train_data_2 = pd.read_csv(
    train_data_2_path,
    encoding="utf-8",
    index_col=False,
    dtype=str,  # 数据类型->字符串
)
print("aaq train_data_2: ", train_data_2.shape)  #  (2602, 2)

# 使用concat函数拼接，axis=0表示垂直方向的拼接，ignore_index=True表示重置索引
train_data = pd.concat([train_data, train_data_2, train_data_qa], axis=0, ignore_index=True)


# ======================已读取所有相似问数据=======================


#去除重复的行
train_data.drop_duplicates(inplace=True)
print("q-q相似句对数据量：", train_data.shape)  #   (103096, 3)

print("清洗数据（耗时操作）...")
train_data['question_content'] = train_data['question_content'].apply(cleaning_func.clean_text)
train_data['question_generate'] = train_data['question_generate'].apply(cleaning_func.clean_text)

print("过滤无意义数据....")
filters = ~(
    (train_data['question_content']=='') # 过滤空数据
    | (train_data['question_content'].str.len() <= 3)  # 过滤长度≤2的数据
    | (train_data['question_content'].str.contains("转人工", na=False))  # 过滤无意义问询
    | (train_data['question_content'].str.contains("试驾活动介绍", na=False))  # 过滤特定内容
    # | (train_data['question_content'].str.contains("OTA升级内容", na=False))  # 过滤特定内容
    | (train_data['question_content'].str.match(r'^\[.*\]$', na=False))  # 过滤答案粘贴
    | (train_data['question_content'].str.match(r'^[a-zA-Z0-9' + r'\s' + punctuation + string.punctuation + r']*$', na=False))  # 过滤全由英文字母、数字、标点组成的数据
)
train_data = train_data[filters]

filters = ~(
    (train_data['question_generate']=='') # 过滤空数据
    | (train_data['question_generate'].str.len() <= 3)  # 过滤长度≤2的数据
    | (train_data['question_content'].str.contains("转人工", na=False))  # 过滤无意义问询
    | (train_data['question_content'].str.contains("试驾活动介绍", na=False))  # 过滤特定内容
    # | (train_data['question_content'].str.contains("OTA升级内容", na=False))  # 过滤特定内容
    | (train_data['question_generate'].str.match(r'^[a-zA-Z0-9' + r'\s' + punctuation + string.punctuation + r']*$', na=False))  # 过滤全由英文字母、数字、标点组成的数据
)
train_data = train_data[filters]
print("所有q-q相似句对数据量：", train_data.shape)  # (102930, 2)

# # 过滤不在知识库中的qq（大可不必）
# train_data = train_data[train_data['question_content'].isin(know_list)]

print("清洗数据（耗时操作）...")
train_data['question_content'] = train_data['question_content'].apply(cleaning_func.clean_text)
train_data['question_generate'] = train_data['question_generate'].apply(cleaning_func.clean_text)


print("结果写入 CSV 文件")
train_data = train_data.rename(columns={'question_content': 'standard_question', 'question_generate': 'similar_question'})
train_data.to_csv(proc_know_path, index=False)

