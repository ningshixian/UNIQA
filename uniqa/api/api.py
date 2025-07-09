import os
import sys
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.getcwd())
sys.path.append(os.path.abspath('.'))

from fastapi import FastAPI, HTTPException
from fastapi import status
from fastapi.responses import JSONResponse, Response
from fastapi import APIRouter, Depends, File, UploadFile, Query

import re
from datetime import datetime
import importlib
import traceback
from pydantic import BaseModel, validator
from starlette.requests import Request
from starlette.testclient import TestClient
import uvicorn
import asyncio
from typing import List, Dict, Optional, Any
from collections import OrderedDict
import json
import requests
import torch
import gc
import pandas as pd

from uniqa.api.request.faq_schema import *
from uniqa.api.response import response_code
from uniqa.logging import Loggers, logDog, log_filter

from uniqa.api.faq import FAQPipeline, DataPreprocessor
from uniqa import Document
from configs.config import *

"""
gunicorn api:app --bind=0.0.0.0:8091 --workers=1 -k uvicorn.workers.UvicornWorker
nohup gunicorn api:router -c configs/gunicorn_config_api.py > logs/api.log 2>&1 &
"""

app = FastAPI(title="FAQ API")

# 初始化组件
preprocessor = DataPreprocessor()
docs = preprocessor.load_data()
faq = FAQPipeline(is_whitening=False)
# faq.load_milvus(docs)


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


# 加载 Milvus（在应用启动时执行）
@app.on_event("startup")
async def startup_event():
    faq.load_milvus(docs)


@app.get("/")
async def read_root():
    return {"message": "Welcome to the FAQ API"}


@app.post("/search")
@log_filter
# async def search_faq(request: Request, item: Item):
def search_faq(request: Request, item: Item):
    """根据输入的查询字符串搜索相关的 FAQ 条目
    """
    item = item.dict()
    query = item["query"]
    top_k = item["top_k"]  # 检索召回数量，默认5
    search_strategy = item["search_strategy"]  # 检索策略（可选 sparse/embedding/hybrid）
    car_type = item["car_type"]         # list[str]=[]
    ota_version = item["ota_version"]   # list[str]=[]  

    # 输入验证
    if not isinstance(top_k, int) or top_k <= 0:
        return response_code.resp_4001(data="topK必须是正整数。")
    if search_strategy not in ['hybrid', 'sparse', 'embedding']:
        return response_code.resp_4001(data="检索策略错误或暂不支持...")
    if not query:
        return response_code.resp_4001(data="输入不能为空")
    
    # # 异常情况处理
    # if isinstance(query, str):
    #     query = [query]
    # elif not isinstance(query, list):
    #     return response_code.resp_4001(data="输入数据类型错误")
    
    # # Do Metadata Filtering
    # faq.filters={   
    #     "operator": "AND",
    #     "conditions": [
    #         # 所有meta.answer的ota_version与输入的ota_version有交集
    #         {
    #             "operator": "OR",
    #             "conditions": [
    #                 {"field": "meta.answer[*].ota_version", "operator": "CONTAINS", "value": ota}
    #                 for ota in ota_version
    #             ],
    #         },
    #         # 所有meta.answer的ota_version都必须大于指定版本
    #         {"field": "meta.answer[*].ota_version", "operator": ">", "value": "6.0"},   
    #         {"field": "meta.valid_time", "operator": ">", "value": str(datetime.now())},
    #         # 所有meta.answer的valid_begin_time要小于当前时间，valid_end_time要大于当前时间
    #         {"field": "meta.answer[*].valid_begin_time", "operator": "<=", "value": datetime.now()},
    #         {"field": "meta.answer[*].valid_end_time", "operator": ">=", "value": datetime.now()},
    #         # 所有meta.answer的car_type必须包含指定车型
    #         {"field": "meta.answer[*].car_type", "operator": "CONTAINS", "value": "理想L6"},
    #         # 过滤掉低置信度结果(可选)
    #         {"field": "score", "operator": ">", "value": 0.3},            
    #     ],
    # },

    # query = list(map(_text_standardize, query))
    results = faq.run(query, top_k=top_k, search_strategy=search_strategy)
    # results = results[:top_k]

    # 将结果转换为可序列化的格式
    serialized_results = []
    for doc in results:
        serialized_results.append({
            "id": doc.id,
            "content": doc.content,
            "score": float(doc.score) if doc.score is not None else None,
            "meta": doc.meta
        })
    
    return response_code.resp_200(data={"query": query, "results": serialized_results})


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
    Loggers.init_config()
    server.run()
    