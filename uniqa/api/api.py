"""
# Uvicorn 为单进程的 ASGI server ，而 Gunicorn 是管理运行多个 Uvicorn ，以达到并发与并行的最好效果。
CUDA_VISIBLE_DEVICES=3 python api.py 
gunicorn api:router --bind=0.0.0.0:8091 --workers=1 -k uvicorn.workers.UvicornWorker
nohup gunicorn api:router -c configs/gunicorn_config_api.py > logs/api.log 2>&1 &
"""

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from fastapi import FastAPI
from fastapi import status
from fastapi.responses import JSONResponse, Response
from fastapi import APIRouter, Depends, File, UploadFile, Query

import re
import time
import importlib
import traceback
from pydantic import BaseModel, validator
from starlette.requests import Request
from starlette.testclient import TestClient
import uvicorn
import asyncio
from typing import List, Dict, Optional, Any
from collections import OrderedDict
import jionlp
import json
import requests
import torch
import gc
import pandas as pd

from uniqa.api.request.faq_schema import *
from uniqa.api.response import response_code
from uniqa.logging import logDog, log_filter

from uniqa.api._faq import FAQ
from uniqa.components.preprocessors import TextCleaner
from configs.config import *

# router = APIRouter()
app = FastAPI(
    title="CallCenter Knowledge Semantic Search",
    description="基于售后知识平台，进行语义检索，召回相似问题",
)

SOURCE_HIGH_PRIORITY = [1, 2]
SOURCE_LOW_PRIORITY = [4, 5]
lib_name2id = {
    "知识库":1,
    "一触即达":2,
    "自定义寒暄库":4,
    "内置寒暄库":5,
}
entity_id2name = {
    "0": ("prebuildEntity", "系统实体"), 
    "1": ("enumerateEntity", "枚举实体"), 
    "2": ("reEntity", "正则实体"), 
    "3": ("intentEntity", "意图实体"), 
    "4": ("otherEntity", "其他实体"), 
    "5": ("algorithmEntity", "理想算法实体"), 
    "": ("?", "未知实体"), 
}

topK = 5    # 检索返回知识的数量，默认5
is_whitening = True
h_score, l_score = 0.88, 0.44  # 0.6 0.44 0.35
# 大于高阈值→直出答案，小于低阈值→澄清话术，二者之间→出列表

# 全局变量声明
global_sentences: List[str] = []
global_qid_dict: Dict = {}
global_sen2qid: OrderedDict = OrderedDict()

# 加载问答系统类
# qa_path_list = [qa_qy_onetouch, qa_custom_greeting, qa_global_greeting]
qa_path_list = [qa4api_path, qa_qy_onetouch, qa_custom_greeting, qa_global_greeting]
faq_sys = FAQ(qa_path_list)

# 加载实体识别模块
from uniqa.components.extractors.custom_ner import EntityExtractor
uie = EntityExtractor()

# 加载文本清理模块
cleaner = TextCleaner(
    convert_to_lowercase=True, 
    remove_punctuation=True, 
    remove_numbers=True, 
    remove_emoji=True, 
    http_normalization=True, 
    phone_normalization= True, 
    time_normalization= True
)


def get_onetouch_dict():
    with open(intent_path, 'r') as f:
        file_contents = f.read()
        intent_dict = json.loads(file_contents)
    with open(param_path, 'r') as f:
        file_contents = f.read()
        param_dict = json.loads(file_contents)
    with open(entity_path, 'r') as f:
        file_contents = f.read()
        entity_dict = json.loads(file_contents)
    return intent_dict, param_dict, entity_dict


intent_dict, param_dict, entity_dict = get_onetouch_dict()


def torch_gc():
    """释放PyTorch的GPU缓存"""
    try:
        # 检查是否有可用的CUDA设备
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # 清空CUDA缓存
            torch.cuda.ipc_collect()  # 收集CUDA内存碎片
        print(f"gpu内存清理完成！")
        # gc.collect()
        # print(f"垃圾回收完成！")
    except Exception as e:
        print(f"清理内存时出错: {e}")


# query查询改写，比如敏感词过滤，错别字纠错，停用词过滤等
def _text_standardize(query):
    query = cleaner.run(texts=[query])[0]

    # 补充：去除文本中的异常字符、冗余字符、HTML标签、括号信息、URL、E-mail、电话号码，全角字母数字转换为半角
    query = jionlp.clean_text(
        text=query,
        remove_html_tag=True,   # HTML标签
        convert_full2half=True, # 全角字母数字转换为半角
        remove_exception_char=False, 
        # remove_exception_char=True, # 删除文本中异常字符，主要保留汉字、常用的标点，单位计算符号，字母数字等
        remove_url=True,
        remove_email=True, 
        remove_redundant_char=True, # 删除文本中冗余重复字符
        remove_parentheses=False,    # 删除括号内容 ✖
        remove_phone_number=False,
    )
    return query.strip()


def validation(query):
    pass


def find_first_knowledge(x: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    获取第一个高优先级的检索结果,并返回该字典
    """
    for item in x:
        if item.get("source") in SOURCE_HIGH_PRIORITY:
            return item
    return {"score": 0}


def classify_answer_type(score, h_score, l_score):
    """
    根据得分分类答案类型。
    """
    if score >= h_score:
        return 1  # 精准匹配
    elif h_score > score >= l_score:
        return 3  # 模糊匹配
    else:
        return 0  # 不回复


@app.post("/ner", summary="算法实体抽取", name="算法实体抽取")
def ner(request: Request, item_ner: Item4ner):
    item_ner = item_ner.model_dump()
    text = [item_ner["text"]]   # list[0]
    entity_name = item_ner["entity_name"]   # 实体名称,对应七鱼的entityType
    param_name = item_ner["param_name"]     # 变量名称，与实体名称一一对应

    # entName2vocab = {}
    # for k,v in entity_dict.items():
    #     if v["entityType"]==5:
    #         entName2vocab[v['entityName']] = v['vocab']
    # print(entName2vocab)

    flag, entity_value, start_idx, end_idx = uie.entity_extract(
        text, 
        entity_mode="algorithmEntity", 
        entity={
            "entityName": entity_name,
            "entityType": 5,
            "vocab": {}
    })

    match_entity = []
    if flag:
        match_entity.append({
            'start': start_idx,  #实体位置
            'end': end_idx,  #实体位置
            'entity_name': entity_name,  #实体名称,对应七鱼的entityType
            'entity_value': entity_value,  #实体值
            'entity_mode': "algorithmEntity",  #实体类型（0-系统/1-枚举/2-正则/3-意图/4-其他/5-算法）
            'param_name': param_name,  #变量名称，与实体名称一一对应
            # 'flow_name': intent.get('flowName', '')  #关联流程名称
            "prebuild": False,  # 是否系统预置实体
            "is_prebuild": False  # 是否系统预置实体
        })
    return response_code.resp_200(data=match_entity)


@app.post("/similar", summary="相似度计算", name="相似度计算")
def similar(request: Request, item_sim: Item4sim):
    item_sim = item_sim.model_dump()
    text1 = item_sim["text1"]
    text2 = item_sim["text2"]
    # 异常情况处理
    if not text1 or not text2:
        return response_code.resp_4001(data="输入不能为空")
    score = faq_sys.recall_module.faiss.cal_sim(text1, text2)
    # 要保证接口的输出是字典格式，且避免 float 类型数据
    return response_code.resp_200(data={'cosine_similarity': str(score[0])})


@app.post("/embedding", summary="转换向量", name="转换向量")
def embedding(request: Request, item_emb: Item4emb):
    item_emb = item_emb.model_dump()
    text = item_emb["text"]

    # 异常情况处理
    if isinstance(text, str):
        text = [text]
    # elif not isinstance(text, list):
    #     return response_code.resp_4001(data="输入数据类型错误")

    vecs = faq_sys.recall_module.faiss.get_vecs(text)

    # JSON格式 不直接支持 NumPy 数组,无法直接将 NumPy数组 直接序列化为 JSON 格式.
    return response_code.resp_200(data={'embedding': json.dumps(vecs.tolist())})


@app.post("/predict")
async def predict_api(request: Request, item: Item):
    """请求 faiss 检索服务，获取相关知识推荐"""
    item = item.dict()
    text = item["text"]
    session_id = item["session_id"]
    topK = item["top_n"]  # 检索召回数量，默认5
    search_strategy = item["search_strategy"]  # 检索策略（可选 bm25/embedding/hybrid）
    
    # 输入验证
    if not isinstance(topK, int) or topK <= 0:
        return response_code.resp_4001(data="topK必须是正整数。")
    if search_strategy not in ['hybrid', 'bm25', 'embedding']:
        return response_code.resp_4001(data="检索策略错误或暂不支持...")
    if not text:
        return response_code.resp_4001(data="输入不能为空")
    
    # 异常情况处理
    if isinstance(text, str):
        text = [text]
    elif not isinstance(text, list):
        return response_code.resp_4001(data="输入数据类型错误")
    
    # 数据清洗
    text = list(map(_text_standardize, text))

    # 再次检查处理后的text
    if not text or not all(text):
        return response_code.resp_5002(data="内部验证数据错误，请重新输入您的问题!")

    # 知识检索，取返回list的第一个结果
    results = faq_sys.search(text, size=topK, search_strategy=search_strategy)

    # 过滤掉低置信度结果(可选)
    results = [result for result in results if result["score"] > 0.3]
    # 保留topK
    results = results[:topK]

    # # 返回第一个检索结果
    # results = results[0][:topK]
    # print("Top1答案：" + results[0]["answer"])
    # for x in results[:topK]:
    #     print(f"{x['score']:.4f}\t{x['standard_sentence']}")
    
    return response_code.resp_200(data={"text": text, "results": results})


@app.post("/predict4cc", summary="FAQ检索式问答", name="FAQ检索")
@log_filter
def predict_api2(request: Request, item2: Item4cc):
    """请求 faiss 检索服务，获取相关知识"""
    item2 = item2.dict()
    text = item2["text"] # 原始/改写后的文本
    topK = item2["top_n"]  # 检索召回数量，默认5
    search_strategy = item2["search_strategy"]  # 检索策略（可选 bm25/embedding/hybrid）
    query_extend = item2["query_extend"]    # 是否进行查询扩展，默认 False
    # library_team = item2["library_team"]  # 检索库id
    exclude_team = item2["exclude_team"]  # 排除的检索库id，1: "FAQ知识库", 2: "一触即达意图", 4: "自定义寒暄", 5: "内置寒暄库",
    car_type = item2["car_type"]  # 车型
    # car_type = car_type[0] if car_type else '' # 暂时取第一个

    # 响应参数
    response_data = {
        "text": text,
        "origin_text": text,
        "answer": "",   # 传给DM用于前端展示
        "answer_type": -1,   # top1回复类型 0：空 1：精准匹配 3：模糊匹配
        "confidence": -1,  # top1 score
        "threshold": {"high": h_score, "low": l_score},  # 高/低阈值
        "detail_results": [],  # 向量检索的所有结果
    }

    # 输入验证
    if not text:
        response_data['answer'] = "输入不能为空"
        return response_code.resp_4001(data=response_data)
    if search_strategy not in ['hybrid', 'bm25', 'embedding']:
        response_data['answer'] = "检索策略错误或暂不支持..."
        return response_code.resp_4001(data=response_data)
    if not isinstance(topK, int) or topK <= 0:
        response_data['answer'] = "topK必须是正整数"
        return response_code.resp_4001(data=response_data)
    
    # 异常情况处理
    if isinstance(text, str):
        text = [text]
    elif not isinstance(text, list):
        response_data['answer'] = "输入数据类型错误"
        return response_code.resp_4001(data=response_data)
    
    # 数据清洗
    text = list(map(_text_standardize, text))

    # 再次检查处理后的输入
    if not text or not all(text):
        response_data['answer'] = "数据验证不通过-无效数据，请重新输入您的问题!"
        return response_code.resp_4001(data=response_data)
    
    # TODO:查询扩展
    # https://tech.meituan.com/2022/02/17/exploration-and-practice-of-query-rewriting-in-meituan-search.html
    if query_extend:
        pass

    # 知识检索，取返回list的第一个结果
    results = faq_sys.search(text, size=topK, search_strategy=search_strategy)

    # 库名转ID
    for result in results:
        result['source'] = lib_name2id.get(result["source"], -1)

    # 排除库检索结果
    if exclude_team:
        logDog.info(f"排除库过滤")
        results = [result for result in results if result["source"] not in exclude_team]

    # 根据车型进行过滤(空→与车型无关进行推荐)
    if car_type:
        logDog.info(f"车型过滤")
        # 车型标签2.0优化-0909：对 query 中提及的车型进行抽取，更新car_type参数
        flag, entity_value, start_idx, end_idx = uie.entity_extract(text, "algorithmEntity", {})
        if flag:
            car_type.append(entity_value)
        pattern = re.compile(r'[_\s]')  # 去除‘车型’里的空格和下划线
        processed_car_types = {pattern.sub('', car.lower()) for car in car_type}
        # 使用列表推导式过滤结果
        results = [
            result for result in results
            if not result.get("car_type") or 
            any([car in pattern.sub('', result["car_type"].lower()) for car in processed_car_types])
        ]
    
    # 检索结果为空，直接返回
    if not results:
        response_data['answer'] = "未匹配到答案，检索结果为空！"
        return response_code.resp_200(data=response_data)

    """
    匹配优先逻辑：
    - 【FAQ + 一键触达意图】是最高优先级
    - 【自定义和内置寒暄库】次高优先级
    - 等级的划分会通过分数来区分
    具体来说：
        1、精准匹配的情况下，
        如果完全匹配命中，直出答案
        如果 top1 score >= 0.88，且 top1 问题属于意图，走一键触达；
        如果 top1 score >= 0.88，且 top1 问题属于知识，走 faq 流程 + （你还想问以下问题...）；

        2、【FAQ + 一键触达意图】都未匹配的情况下，那就属于模糊匹配
        如果 0.88 > top1 score >= 0.6
            - 当寒暄库top分数特别高（比如 0.95 或 两边的差值达到一定程度），优先出寒暄库
            - 否则，出检索的 top5 faq/意图结果列表
        如果 top1 score < 0.6
            - 当寒暄库top分数特别高（比如 0.95 或 两边的差值达到一定程度），优先出寒暄库
            - 否则，出澄清话术
    """
    # 获取第一个高优先级的检索结果
    first_know = find_first_knowledge(results)
    # 如果top1（score > 0.6 & 次高优先级 & 高于first_know的得分 0.1）则优先出寒暄库
    if results[0]['source'] in SOURCE_LOW_PRIORITY and results[0]['score'] > l_score and (results[0]['score'] - first_know['score'] > 0.1):
        results = [result for result in results if result["source"] in SOURCE_LOW_PRIORITY]
    else:   # 否则，不考虑寒暄库
        results = [result for result in results if result["source"] in SOURCE_HIGH_PRIORITY]

    # 检索结果为空，直接返回
    if not results:
        response_data['answer'] = "未匹配到答案，检索结果为空！"
        return response_code.resp_200(data=response_data)

    # 保留topK
    results = results[:topK]
    # print(results)

    # TODO：匹配一触即达
    intentname2id = {}
    for k,v in intent_dict.items():
        intentname2id[v['intentName']] = k
        intentname2id[v['primaryName']] = k
    for r in results:
        r['match_entity'] = []  #匹配实体列表
        if r['source'] == 2 and r['score'] >= l_score:    # 一触即达
            logDog.info(f"匹配到一触即达意图，进行实体提取")
            intent_id = intentname2id.get(r['standard_sentence'])
            intent = intent_dict.get(intent_id, {})
            for param_id in intent.get('paramIdList', []):
                param = param_dict.get(param_id, {})
                entity_id = param.get('entityId', '')
                entity = entity_dict.get(entity_id, {})
                entity_name = entity.get('entityName', '')
                entity_mode = entity_id2name[str(entity.get('entityType', ''))][0] # 实体类型（0-系统/1-枚举/2-正则/3-意图/4-其他/5-算法）
                
                # 提取实体
                # flag 表示是否实体匹配成功
                flag, entity_value, start_idx, end_idx = uie.entity_extract(text, entity_mode, entity)
                logDog.info(f"实体提取Flag: {flag}, 实体名称：{entity_name}, 实体类型：{entity_mode}, 实体值：{entity_value}, 实体位置：{start_idx, end_idx}")

                if flag:
                    detail_entity = {
                        'start': start_idx,  #实体位置
                        'end': end_idx,  #实体位置
                        'entity_name': entity_name,  #实体名称 entityType
                        'entity_value': entity_value,  #实体值
                        'entity_mode': entity_mode,  #实体类型（0-系统/1-枚举/2-正则/3-意图/4-其他/5-算法）
                        'param_name': param.get('paramName', ''),  #关联变量
                        # 'flow_name': intent.get('flowName', '')  #关联流程名称
                        "prebuild": False,  # 是否系统预置实体
                        "is_prebuild": False  # 是否系统预置实体
                    }
                    r['match_entity'].append(detail_entity)

    # 调试输出格式
    answer = "\n".join(
        [f"{x['score']:.4f}\t{x['standard_sentence']}" for x in results]
    )
    answer += "\n\n" + "Top1答案：" + results[0].get("answer", "")  # IndexError: list index out of range
    # 答案类型（针对 top1）
    answer_type = classify_answer_type(results[0]['score'], h_score, l_score)
    # 填充response_data
    response_data['text'] = text[0]
    response_data['answer'] = answer
    response_data['answer_type'] = answer_type
    response_data['confidence'] = results[0]['score']
    response_data['detail_results'] = results
    return response_code.resp_200(data=response_data)


@app.get("/update_faq")
def update_faq(request: Request):
    """
    更新全局变量、重新加载Faiss和BM25索引
    重新初始化了模型，虽然最后任务结束，但是并不会释放显存，最终会显存溢出，考虑 del+gc
    https://wjwsm.top/2023/06/11/NLP%E6%A8%A1%E5%9E%8B%E9%83%A8%E7%BD%B2%E6%80%BB%E7%BB%93(fastapi+uvicorn)/
    """
    try:
        logDog.info("开始全量更新FAQ系统...")

        # 重新导入模块
        from uniqa.components import recall
        importlib.reload(recall)
        # 重新创建召回模块实例
        faq_sys.recall_module = recall.Recall(qa_path_list) 

        # 清空原有全局变量
        global global_sentences, global_qid_dict, global_sen2qid
        global_sentences = []
        global_qid_dict = {}
        global_sen2qid = OrderedDict()

        # 更新UIE模块
        global uie
        from uniqa.components.extractors.custom_ner import EntityExtractor
        uie = EntityExtractor()

        # 更新意图和实体字典
        global intent_dict, param_dict, entity_dict
        intent_dict, param_dict, entity_dict = get_onetouch_dict()

        torch_gc()  # 随着更新次数增多，显存占用会变大，所以顶一个 torch_gc() 方法完成对显存的回收

        logDog.info("FAQ系统全量更新成功\n")
        return {'status': True, 'msg': 'Successfully updated all FAQ components'}
    except Exception as e:
        error_msg = f"【update_api接口】更新FAQ系统时发生错误: {str(e)}"
        logDog.error(f"{error_msg}\n{traceback.format_exc()}")
        return {'status': False, 'msg': error_msg}


@app.get("/update_faq_onetouch")
def update_faq_onetouch(request: Request):
    """
    增量更新FAQ中的一触即达知识
    qa_qy_onetouch 被覆盖更新
    """
    try:
        logDog.info("开始更新FAQ一触即达...")

        from uniqa.components import recall
        # 上面重新加载faq_sys.recall_module会导致update_faq_know中的索引更新失效
        # 如果修改为仅更新现有实例的数据，那还得做 diff，找到被修改的内容后再能准确索引，太麻烦！
        faq_sys.recall_module = recall.Recall(qa_path_list) 

        # 与已有的增量知识集成
        if global_sentences:
            logDog.info(f"合并{len(global_sentences)}条增量索引数据")
            # 更新词典和映射
            faq_sys.recall_module.faiss.qid_dict.update(global_qid_dict)
            faq_sys.recall_module.faiss.sen2qid.update(global_sen2qid)
            faq_sys.recall_module.faiss.sentences.extend(global_sentences)
            # 生成并添加新向量
            new_vecs = faq_sys.recall_module.faiss.get_vecs(global_sentences)
            new_vecs = faq_sys.recall_module.faiss.__tofloat32__(new_vecs)
            faq_sys.recall_module.faiss.index.add(new_vecs)
            
        # 更新UIE模块
        global uie
        from uniqa.components.extractors.custom_ner import EntityExtractor
        uie = EntityExtractor()

        # 更新意图和实体字典
        global intent_dict, param_dict, entity_dict
        intent_dict, param_dict, entity_dict = get_onetouch_dict()

        torch_gc()  # 随着更新次数增多，显存占用会变大，所以顶一个 torch_gc() 方法完成对显存的回收

        logDog.info("FAQ一触即达全量更新成功\n")
        return {'status': True, 'msg': 'Successfully updated FAQ onetouch'}
    except Exception as e:
        error_msg = f"【update_faq_onetouch接口】更新FAQ一触即达时发生错误: {str(e)}"
        logDog.error(f"{error_msg}\n{traceback.format_exc()}")
        return {'status': False, 'msg': error_msg}


@app.get("/update_faq_know/{item_id}")
def update_faq_know(item_id: int):
    """
    增量更新FAQ知识库
    Args:
        item_id (int): 更新类型 - 1: 标准问增量更新, 2: 相似问增量更新
    """
    try:
        file_path = None
        if item_id == 1:
            # 知识标准问增量同步存储
            file_path = './data_factory/update_robot_know.csv'
        elif item_id == 2:
            # 相似问增量同步存储
            file_path = './data_factory/update_robot_know_sim.csv'
        else:
            error_msg = 'Invalid item_id, must be 1 or 2'
            logDog.error(error_msg)
            return {'status': False, 'msg': error_msg}
        
        logDog.info(f"开始 know={item_id} 知识增量更新")
        
        # 读取CSV文件
        # 空值都是默认为 NAN，设置 keep_default_na=False 让读取出来的空值是空字符串
        df = pd.read_csv(file_path, encoding="utf-8", keep_default_na=False)

        if df.empty:
            logDog.info(f"文件 {file_path} 为空，无需更新")
            return {'status': True, 'msg': 'No data to update'}

        # 收集新句子
        new_sentences = []
        new_qid_dict = {}
        new_sen2qid = OrderedDict()

        # 处理标准问更新
        if item_id==1:
            for i,row in df.iterrows():
                question_id = row["question_id"]
                question_content = row["question_content"]
                answer_content = row["answer_content"]
                car_type = row["car_type"]

                # 获取现有相似问（如果有）
                similar_sentence = faq_sys.recall_module.faiss.qid_dict.get(
                    question_id, {}).get("similar_sentence", [])

                # 更新问题-答案字典
                new_qid_dict[question_id] = {
                    'standard_sentence': question_content,
                    'similar_sentence': similar_sentence,
                    'answer': answer_content,
                    'source': "知识库",
                    'car_type': car_type
                }
                # 更新问题-ID映射和句子列表
                new_sen2qid[question_content] = question_id
                new_sentences.append(question_content)

        # 处理相似问更新
        if item_id==2:
            for i, row in df.iterrows():
                question_id = row["question_id"]
                similar_question = row["similar_question"]
                # 检查question_id是否在现有词典中
                if question_id not in faq_sys.recall_module.faiss.qid_dict:
                    logDog.warning(f"标准问ID {question_id} 不存在，相似问 '{similar_question}' 将使用占位标准问")
                        
                # 获取现有数据
                question_content = faq_sys.recall_module.faiss.qid_dict.get(
                    question_id, {}).get("standard_sentence", "标准问插入延迟")
                similar_sentence = faq_sys.recall_module.faiss.qid_dict.get(
                    question_id, {}).get("similar_sentence", []).copy()  # 创建副本以避免修改原始数据
                answer_content = faq_sys.recall_module.faiss.qid_dict.get(
                    question_id, {}).get("answer_content", "")
                car_type = faq_sys.recall_module.faiss.qid_dict.get(
                    question_id, {}).get("car_type", "")
                
                # 添加新的相似问
                similar_sentence.append(similar_question)

                # 更新问题-答案字典
                new_qid_dict[question_id] = {
                    'standard_sentence': question_content,
                    'similar_sentence': similar_sentence,
                    'answer': answer_content,
                    'source': "知识库",
                    'car_type': car_type
                }
                # 更新问题-ID映射和句子列表
                new_sen2qid[similar_question] = question_id
                new_sentences.append(similar_question)

        # 如果有新句子需要添加到当前索引
        if new_sentences:
            logDog.info(f"添加 {len(new_sentences)} 条新句子到索引")
            # 更新词典和映射
            faq_sys.recall_module.faiss.qid_dict.update(new_qid_dict)
            faq_sys.recall_module.faiss.sen2qid.update(new_sen2qid)
            faq_sys.recall_module.faiss.sentences.extend(new_sentences)
            # 生成并添加新向量
            new_vecs = faq_sys.recall_module.faiss.get_vecs(new_sentences)
            new_vecs = faq_sys.recall_module.faiss.__tofloat32__(new_vecs)
            faq_sys.recall_module.faiss.index.add(new_vecs)

            # 更新全局变量
            global global_sentences, global_qid_dict, global_sen2qid
            global_sentences.extend(new_sentences)
            global_qid_dict.update(new_qid_dict)
            global_sen2qid.update(new_sen2qid)

        torch_gc()  # 随着更新次数增多，显存占用会变大，所以顶一个 torch_gc() 方法完成对显存的回收

        logDog.info(f"know={item_id} 知识增量更新成功\n")
        return {'status': True, 'msg': f'Successfully updated {len(new_sentences)} entries'}
    except Exception as e:
        error_msg = f"【update_faq_know接口】know={item_id} 增量更新索引时发生错误: {str(e)}"
        logDog.error(f"{error_msg}\n{traceback.format_exc()}")
        return {'status': False, 'msg': error_msg}


# def merge_dicts(dict_a, dict_b):
#     result = dict_a.copy()  # 创建 A 的副本
#     result.update(dict_b)   # 使用 B 更新结果，冲突时以 B 为准
#     return result


if __name__ == "__main__":
    config = uvicorn.Config(
        "api:app", 
        host='0.0.0.0', port=8091, 
        workers=1, 
        # limit_concurrency=200,  # 最大并发数，默认 None
        # reload=True,
        # debug=True,
    )
    server = uvicorn.Server(config)
    # 将uvicorn输出的全部让loguru管理
    logDog.init_config()
    server.run()
    