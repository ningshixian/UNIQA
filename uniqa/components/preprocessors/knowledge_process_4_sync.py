import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from zhon.hanzi import punctuation
import string
import re
import copy
import sys
sys.path.append(r"../")
from uniqa.components.preprocessors import cleaning_func

"""
【知识数据】清洗&处理脚本
- 确保数据集中的问题和答案都是清晰、有意义且唯一的
- 知识数据获取SQL：https://li.feishu.cn/wiki/WZ44wI5lJixNvLkHtR5c8zEwnwb?fromScene=spaceOverview
- 输出文件：qa4api.csv、qa4rec.csv、processed_crm_knowledge_pair.csv
"""

# 需要处理的知识数据
# extract_qa_data_path = 'crm_qa_llm抽取_智能驾舱.csv'
knowledge_robot_path = '../data_factory/crm_knowledge_robot.csv'  # 用于faq
knowledge_common_path = '../data_factory/crm_knowledge_common.csv'  # 用于知识推荐
car_type_path = '../data_factory/crm_knowledge_car_label.csv'
knowledge_similarQ_path = '../data_factory/crm_knowledge_similarQ.csv'
# 输出文件
result_1_path = "../data_factory/processed_crm_knowledge_pair.csv"  # 用于评测
result_2_path = "../data_factory/qa4api.csv"  # 用于faq
result_3_path = "../data_factory/qa4rec.csv"  # 用于知识推荐

# 读取问题车型标签表
# question_id,vehicle_series_name,vehicle_model_name
car_type_data = pd.read_csv(
    car_type_path,
    encoding="utf-8",
    index_col=False,
)
car_type_dict = {}
for i,row in car_type_data.iterrows():
    car_type_dict.setdefault(row['question_id'], "")
    car_type_dict[row['question_id']] += f"{row['vehicle_series_name']} {row['vehicle_model_name']},"

# 读取相似问知识数据
# ,question_id,similar_question
knowledge_simQ = pd.read_csv(
    knowledge_similarQ_path,
    encoding="utf-8",
    index_col=False,
)
qid2sim = {}
for i,row in knowledge_simQ.iterrows():
    qid2sim.setdefault(row['question_id'], [])
    qid2sim[row['question_id']].append(row['similar_question'])

# 读取qa知识数据
# question_id,question_content,answer_content,base_name
know_data = pd.read_csv(
    knowledge_robot_path,
    encoding="utf-8",
    index_col=False,
    # dtype=str,  # 数据类型->字符串
)
know_data = know_data[['question_id', 'question_content', 'answer_content', 'base_name']]

know_data4rec = pd.read_csv(
    knowledge_common_path,
    encoding="utf-8",
    index_col=False,
    # dtype=str,  # 数据类型->字符串
)
know_data4rec = know_data4rec[['question_id', 'question_content', 'answer_content', 'base_name']]


# # 添加质检之后的llm抽取QA数据 2024.03.28
# extract_data = pd.read_csv(
#     extract_qa_data_path,
#     encoding="utf-8",
#     index_col=False,
#     # dtype=str,  # 数据类型->字符串
# )
# extract_data = extract_data[extract_data['是否抽取准确']=='是']
# extract_data = extract_data[['qestion', 'anwser']]
# extract_data['base_name'] = 'LLM抽取'
# extract_data['anwser'] = extract_data['anwser'].str.strip(punctuation + string.punctuation + "\n")
# # 在答案前添加了【LLM】标签以区分来源
# extract_data['anwser'] = extract_data['anwser'].apply(lambda x: "【LLM】"+x)
# extract_data.rename(columns={'qestion': 'question_content', 'anwser': 'answer_content'}, inplace=True)
# print(f"【LLM】抽取qa数据量：{extract_data.shape}\n")  # (2981, 3)
# # 使用concat函数拼接，axis=0表示垂直方向的拼接，ignore_index=True表示重置索引
# know_data = pd.concat([know_data, extract_data], axis=0, ignore_index=True)

# 添加车型标签
know_data['car_type'] = know_data['question_id'].map(car_type_dict)
know_data4rec['car_type'] = know_data4rec['question_id'].map(car_type_dict)


def clean_knowledge_data(know_data: pd.DataFrame) -> pd.DataFrame:
    
    # 定义一个判断字符串是否被包含的函数
    def is_contained(s1, s2):
        s1 = s1[:-1]
        s2 = s2[:-1]
        return s1 in s2 or s2 in s1

    #去除重复的行
    know_data.drop_duplicates(inplace=True)
    # print(know_data.head(5))
    print("【知识】qa数据量：", know_data.shape)  #  (11721, 5)->(8945, 5)->(1266, 5)

    print("清洗数据（耗时操作）...")
    know_data['question_content'] = know_data['question_content'].apply(cleaning_func.clean_text)
    # know_data['answer_content'] = know_data['answer_content'].apply(data_cleaning.clean_text)

    print("过滤无意义数据....")
    filters = ~(
        (know_data['question_content']=='') # 过滤空数据
        | (know_data['question_content'].str.len() <= 2)  # 过滤长度≤2的数据
        | (know_data['question_content'].str.contains("转人工", na=False))  # 过滤无意义问询
        | (know_data['question_content'].str.contains("试驾活动介绍", na=False))  # 过滤特定内容
        | (know_data['question_content'].str.contains("OTA升级内容", na=False))  # 过滤特定内容
        | (know_data['question_content'].str.match(r'^\[.*\]$', na=False))  # 过滤答案粘贴
        | (know_data['question_content'].str.match(r'^[a-zA-Z0-9' + r'\s' + punctuation + string.punctuation + r']*$', na=False))  # 过滤全由英文字母、数字、标点组成的数据
    )
    know_data = know_data[filters]

    # print("过滤答案冗余数据....")
    # know_data = know_data.sort_values('question_content')
    # # 过滤答案相同的知识，保留第一个（有太多答案相同的知识了，如“座椅加热怎么....”）
    # know_data.drop_duplicates(subset='answer_content', keep='first', inplace=True)

    # print("过滤相同标准问...")
    # # 【去除首尾标点】后，过滤相同的标准问，只保留第一个（由于存在问题相同，适用车型不同的问题，pass）
    # know_data['q_new'] = know_data['question_content'].str.strip(punctuation + string.punctuation)
    # know_data.drop_duplicates(subset='q_new', keep='first', inplace=True)
    # know_data.drop(columns='q_new', inplace=True)

    # # 去除其中【问题】列值存在包含关系的行，保留第一个（很多“MEGA”问题，需要保留）
    # is_duplicate = pd.Series(False, index=know_data.index)
    # for i in range(1, len(know_data)):
    #     is_duplicate.iloc[i] = is_contained(know_data.loc[i, 'question_content'], know_data.loc[i - 1, 'question_content'])
    # know_data = know_data[~is_duplicate]

    print("清洗数据（耗时操作）...")
    know_data['question_content'] = know_data['question_content'].apply(cleaning_func.clean_text)

    print("【知识】qa筛选后数据量：", know_data.shape)
    return know_data


def clean_knowledge_data4rec(know_data4rec: pd.DataFrame) -> pd.DataFrame:
    #去除重复的行
    know_data4rec.drop_duplicates(inplace=True)
    # print(know_data4rec.head(5))
    print("【推荐】qa数据量：", know_data4rec.shape)  #  (8740, 5)-> (8945, 5)

    # QA数据做筛选 1️⃣包含“级别：二级”；2️⃣包含飞书链接
    filters = ~(
        (know_data4rec['question_content'].str.match(r"(^级别[:：].{2}[。 ]{1,3}.{2}[:：])", na=False))
        | (know_data4rec['question_content'].str.match(r"(^级别[:：].级[。 ])", na=False))
        | (know_data4rec['question_content'].str.match(r"(^流程.*简.[:：])", na=False))
        | (know_data4rec['question_content'].str.findall(r'https?://li.feishu.cn'))
    )
    know_data4rec = know_data4rec[filters]

    print("清洗数据（耗时操作）...")
    know_data4rec['question_content'] = know_data4rec['question_content'].apply(cleaning_func.clean_text)

    print("过滤无意义数据....")
    filters = ~(
        (know_data4rec['question_content']=='') # 过滤空数据
        | (know_data4rec['question_content'].str.len() <= 2)  # 过滤长度≤2的数据
        | (know_data4rec['question_content'].str.contains("转人工", na=False))  # 过滤无意义问询
        | (know_data4rec['question_content'].str.contains("试驾活动介绍", na=False))  # 过滤特定内容
        | (know_data4rec['question_content'].str.contains("OTA升级内容", na=False))  # 过滤特定内容
        | (know_data4rec['question_content'].str.match(r'^\[.*\]$', na=False))  # 过滤答案粘贴
        | (know_data4rec['question_content'].str.match(r'^[a-zA-Z0-9' + r'\s' + punctuation + string.punctuation + r']*$', na=False))  # 过滤全由英文字母、数字、标点组成的数据
    )
    know_data4rec = know_data4rec[filters]

    # print("过滤答案冗余数据....")
    # know_data4rec = know_data4rec.sort_values('question_content')
    # # 过滤答案相同的知识，保留第一个（有太多答案相同的知识了，如“座椅加热怎么....”）
    # know_data4rec.drop_duplicates(subset='answer_content', keep='first', inplace=True)

    print("过滤相同标准问...")
    # 【去除首尾标点】后，过滤相同的标准问，保留第一个（“如何设置按时出发”）
    know_data4rec['q_new'] = know_data4rec['question_content'].str.strip(punctuation + string.punctuation)
    know_data4rec.drop_duplicates(subset='q_new', keep='first', inplace=True)
    know_data4rec.drop(columns='q_new', inplace=True)

    print("清洗数据（耗时操作）...")
    know_data4rec['question_content'] = know_data4rec['question_content'].apply(cleaning_func.clean_text)

    print("【推荐】qa筛选后数据量：", know_data4rec.shape)  # (8129, 5)->(8326, 5)
    return know_data4rec


know_data = clean_knowledge_data(know_data)
know_data['source'] = '知识库'  # 添加数据源
know_data4rec = clean_knowledge_data4rec(know_data4rec)
know_data4rec['source'] = '知识库'  # 添加数据源


# 插入相似问
new_sim1, new_sim2 = [], []
for qid,sim_list in qid2sim.items():
    rows_with_qid = know_data[know_data['question_id'] == qid]
    if not rows_with_qid.empty:
        for sim_q in sim_list:
            rows_to_update = rows_with_qid.values.tolist()[0]
            rows_to_update[1] = str(sim_q)
            rows_to_update[2] = ""  # 相似问清空答案
            new_sim1.append(rows_to_update)
    
    rows_with_qid = know_data4rec[know_data4rec['question_id'] == qid]
    if not rows_with_qid.empty:
        for sim_q in sim_list:
            rows_to_update = rows_with_qid.values.tolist()[0]
            rows_to_update[1] = str(sim_q)
            rows_to_update[2] = ""  # 相似问清空答案
            new_sim2.append(rows_to_update)
# 插入相似问
know_data = pd.concat([know_data, pd.DataFrame(new_sim1, columns=know_data.columns)], ignore_index=True)
know_data4rec = pd.concat([know_data4rec, pd.DataFrame(new_sim2, columns=know_data4rec.columns)], ignore_index=True)
new_sim1, new_sim2 = [], []
# 保持相同ID的行在一起
know_data = know_data.groupby('question_id').apply(lambda x: x).reset_index(drop=True)
know_data4rec = know_data4rec.groupby('question_id').apply(lambda x: x).reset_index(drop=True)
print("插入相似问：", know_data.shape)
print("插入相似问：", know_data4rec.shape)


print("======================插入相似问完成✔=========================")


"""
（可选）将运营问题反馈badcase 添加给 know_data 和 know_data4rec
"""

know_data = know_data[know_data['question_content'] != '车辆长期停放是否会影响车辆？']
new_rows = [
    {"question_content": '怎么开启全场景辅助驾驶（LCC）？（AD Pro）', "question_generate": 'L7也能使用自动驾驶吗？'},
    {"question_content": '理想家用充电桩是否可以转移，是否收费？', "question_generate": '怎么改家用充电桩的位置'},
    {"question_content": '要是长期停放应该如何保证车辆电量？', "question_generate": '车辆长期停放会有什么影响？'},
    {"question_content": '要是长期停放应该如何保证车辆电量？（MEGA）', "question_generate": '车辆长期停放会有什么影响？'},
    {"question_content": '车辆标准质保周期是多久', "question_generate": '轮胎质保期是多久的'},
    {"question_content": '如何查找附近的充电站？', "question_generate": 'mega我们开出去如果没有5c充电头还可以用哪些充电的'},
    {"question_content": '商城购买便携式随车充使用注意事项', "question_generate": '商城的便携式随车充的工作电压是否为220V'},
    {"question_content": 'OPPO Watch 4 X理想汽车商城专供版如何开通eSIM？', "question_generate": 'OPPO手表4哪些地区能开通esim'},
    {"question_content": '智能钥匙更换电池如何操作？', "question_generate": '车钥匙盖怎么打开'},
]
new_rows_dict = {x["question_content"]:x["question_generate"] for x in new_rows}
# 找到know_data中question_content列与new_rows_dict键匹配的行的索引
new_rows_index = know_data[know_data['question_content'].isin(new_rows_dict.keys())].index
# 深 copy这些行，以避免影响原始数据
new_rows_df = know_data.loc[new_rows_index].copy()
# 更新new_rows_df的question_content列，使用new_rows_dict中的值
new_rows_df['question_content'] = new_rows_df['question_content'].apply(lambda x: new_rows_dict[x])
# 初始化answer_content列为空字符串，作为相似问的标识
new_rows_df['answer_content'] = ""
know_data = pd.concat([know_data, new_rows_df], axis=0, ignore_index=True)
know_data4rec = pd.concat([know_data4rec, new_rows_df], axis=0, ignore_index=True)


print("======================（可选）添加运营问题反馈badcase✔=========================")


# know_data = know_data[~know_data['base_name'].isin(['服务专家内部资料库', '多媒体库'])]
print(f"----->{know_data.shape[0]}条结果写入qa4api.csv文件，用于faq知识检索")   # 11203->8430->1224
know_data = know_data.reset_index(drop=True)    # 重置索引
know_data.to_csv(result_2_path) 

# know_data = know_data[~know_data['answer_content'].str.startswith("【LLM】")]
print(f"----->{know_data.shape[0]}条结果写入processed_.csv文件，用于评测") # 8224->8430->1224
know_data = know_data.reset_index(drop=True)    # 重置索引
know_data.to_csv(result_1_path) 


print(f"----->{know_data4rec.shape[0]}条结果写入qa4rec.csv文件，保留5个库+QA筛选，用于知识推荐")  # 8020->8216->8351
know_data4rec = know_data4rec.reset_index(drop=True)    # 重置索引
know_data4rec.to_csv(result_3_path) 
