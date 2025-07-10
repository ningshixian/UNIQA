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
gunicorn uniqa.api.api:app --bind=0.0.0.0:8091 --workers=1 -k uvicorn.workers.UvicornWorker
nohup gunicorn uniqa.api.api:router -c configs/gunicorn_config_api.py > logs/api.log 2>&1 &
"""

app = FastAPI(title="FAQ API")

# 初始化组件
preprocessor = DataPreprocessor()
docs = preprocessor.load_data(data_path="uniqa/data/demo.json")

# Do Metadata Filtering
filters={
    "operator": "AND",
    "conditions": [
        # {"field": "score", "operator": ">", "value": 0.3},    # 过滤掉低置信度结果(非meta数据❌)
        # {"field": "meta.answes[*].status", "operator": "==", "value": 1},   # Milvus 不支持嵌套字段查询❌
        {"field": "meta.status", "operator": "==", "value": 1},  # ✓
        # 过滤过期问题 ✓
        {
            "operator": "OR",
            "conditions": [
                {"field": "meta.valid_begin_time", "operator": "==", "value": None},
                {"field": "meta.valid_begin_time", "operator": "<=", "value": str(datetime.now())},
            ],
        }, 
        {
            "operator": "OR",
            "conditions": [
                {"field": "meta.valid_end_time", "operator": "==", "value": None},
                {"field": "meta.valid_end_time", "operator": ">=", "value": str(datetime.now())},
            ],
        },
        # TEXT_MATCH
    ]
}

# 初始化QA bot
faq = FAQPipeline(is_whitening=False)
faq.setup_milvusDB_retriever(docs, filters)


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


# # 加载 Milvus（在应用启动时执行）
# @app.on_event("startup")
# async def startup_event():
#     faq.setup_milvusDB_retriever(docs, filters)


@app.get("/")
async def read_root():
    return {"message": "Welcome to the FAQ API"}


@app.post("/search")
@log_filter
def search_faq(request: Request, item: Item):
    """根据输入的查询字符串搜索相关的 FAQ 条目
    """
    item = item.dict()
    query = item["query"]   # 如何更新OTA系统 ｜ 怎样升级车载系统 ｜ 软件更新方法
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

    # query = list(map(_text_standardize, query))

    # 第一步：检索召回相关文档
    results = faq.run(query, top_k=top_k, search_strategy=search_strategy)
    # results = results[:top_k]

    # 第二步：复杂filters逻辑 → 筛选答案
    serialized_results = []
    user_conditions = {"car_type": car_type, "ota_version": ota_version}
    for doc in results:
        # 提取该问题的所有答案
        answers = doc.meta.get("answers", [])

        # 计算答案与用户条件的匹配度
        score_list = []
        for ans in answers:
            if ans.get("status") != 1:  # 过滤答案级无效状态
                continue
            match_score = faq._match_conditions(ans, user_conditions)  # 条件匹配度（0~1）
            score_list.append(match_score)
            # # 综合得分 = 问题语义相似度（doc.score） + 条件匹配度（match_score）
            # combined_score = round(doc.score + match_score, 4)

        # 筛选 score_list中大于等于0.7 的index
        index_list = [i for i, score in enumerate(score_list) if score >= 0.7]
        if index_list:
            serialized_results.append(
                {
                    "question": doc.content,  # 匹配的问题
                    "question_id": doc.meta.get("question_id"),
                    "question_type": doc.meta.get("question_type"),
                    "category": doc.meta.get("category", ""), 
                    "is_main_question": doc.meta.get("is_main_question"),
                    "score": round(float(doc.score), 4),    # fastapi 不支持np.float类型
                    "source": 1,
                    "answer": [answers[i] for i in index_list]  # 符合条件的答案
                }
            )
        else:
            # 获取默认答案索引（车型标签、生效时间、最高/最低ota版本都为空的元素）
            default_index = None
            for i, ans in enumerate(answers):
                if not ans.get("car_type") and not ans.get("effective_time") and not ans.get("max_ota_version") and not ans.get("min_ota_version"):
                    default_index = i
                    break
            # 添加默认答案
            if default_index is not None and default_index not in index_list:
                serialized_results.append(
                {
                    "question": doc.content,  # 匹配的问题
                    "question_id": doc.meta.get("question_id"),
                    "question_type": doc.meta.get("question_type"),
                    "category": doc.meta.get("category", ""), 
                    "is_main_question": doc.meta.get("is_main_question"),
                    "score": round(float(doc.score), 4),    # fastapi 不支持np.float类型
                    "source": 1,
                    "answer": answers[default_index]  # 默认答案
                }
            )

    # 将结果转换为可序列化的格式，存储在serialized_results
    # 按综合得分排序，取Top K答案
    serialized_results.sort(key=lambda x: x["score"], reverse=True)
    serialized_results = serialized_results[:top_k]     # 即detail_results
    # logDog.info(serialized_results)
    return response_code.resp_200(data={"query": query, "results": serialized_results})


@app.get("/full_update")
def full_update(request: Request):
    """更新全局变量"""
    try:
        logDog.info("开始全量更新FAQ系统...")

        global faq
        preprocessor = DataPreprocessor()
        docs = preprocessor.load_data(data_path="uniqa/data/demo.json")
        faq = FAQPipeline(is_whitening=False)
        faq.setup_milvusDB_retriever(docs, filters)

        torch_gc()  # 随着更新次数增多，显存占用会变大，所以顶一个 torch_gc() 方法完成对显存的回收

        logDog.info("FAQ系统全量更新成功\n")
        return {'status': True, 'msg': 'Successfully updated all FAQ components'}
    except Exception as e:
        error_msg = f"【full_update 接口】更新FAQ系统时发生错误: {str(e)}"
        logDog.error(f"{error_msg}\n{traceback.format_exc()}")
        return {'status': False, 'msg': error_msg}


@app.get("/incremental_update")
def incremental_update(request: Request):
    """增量更新FAQ知识库 
    统一通过 write_documents 和 delete_documents 实现增删改，而无需每次都重新索引所有数据。
    """
    try:
        # 获取 Milvus 中所有文档的 id 和 meta
        # 注意：对于非常大的数据集，一次性获取所有文档可能效率不高。
        # 更好的方法是分批获取或使用更高级的同步策略（见后文）。
        # 这里为了演示，我们获取全部。
        milvus_docs = faq.milvus_document_store.filter_documents()
        milvus_data_map = {doc.id: doc.meta for doc in milvus_docs}

        # --- 3. 更新和添加 (UPDATE & ADD) ---
        print("\n--- 3. 更新和添加 (UPDATE & ADD) ---")

        # 模拟数据更新：答案内容变了，并且增加了一个新的相似问题
        update_and_add_docs = preprocessor.load_data("uniqa/data/demo2.json")
        print(f"从更新后的 JSON 生成了 {len(update_and_add_docs)} 个 Haystack Document:")
        for doc in update_and_add_docs:
            print(f"  - ID: {doc.id}, Content: '{doc.content}'")

        # 嵌入并使用 OVERWRITE 策略写入
        from uniqa.document_stores.types import DuplicatePolicy
        embedded_update_docs = faq.doc_embedder.run(documents=update_and_add_docs)["documents"]
        # update_documents
        count = faq.milvus_document_store.write_documents(
            embedded_update_docs,
            policy=DuplicatePolicy.OVERWRITE
        )

        # print(f"\n增量更新操作写入了 {count} 份文档。")
        # print(f"当前 DocumentStore 中的文档总数: {faq.milvus_document_store.count_documents()}")

        # # 验证更新结果
        # print("\n🔍 验证更新结果:")
        # # 检查主问题文档是否被更新
        # updated_doc_kb001 = faq.milvus_document_store.filter_documents(
        #     filters={"operator": "AND", "conditions": [{"field": "id", "operator": "==", "value": "KBXX1"}] }
        # )[0]
        # print(f"ID 'KBXX1' 的答案已更新: '{updated_doc_kb001.meta['answers'][0]['answerContent']}'")
        # # 检查新添加的相似问题文档是否存在
        # new_doc_kb001_3 = faq.milvus_document_store.filter_documents(
        #     filters={"operator": "AND", "conditions": [{"field": "id", "operator": "==", "value": "KBXX1_4"}] }
        # )
        # if new_doc_kb001_3:
        #     print("ID 'KBXX1_4' 的新相似问题已成功添加。")
        # else:
        #     print("错误：新相似问题添加失败！")

        torch_gc()  # 随着更新次数增多，显存占用会变大，所以顶一个 torch_gc() 方法完成对显存的回收

        logDog.info(f"\n增量更新操作写入了 {count} 份文档。")
        logDog.info(f"当前 DocumentStore 中的文档总数: {faq.milvus_document_store.count_documents()}")
        return {'status': True, 'msg': f'Successfully updated {count} entries'}
    
    except Exception as e:
        error_msg = f"【incremental_update 接口】 增量更新索引时发生错误: {str(e)}"
        logDog.error(f"{error_msg}\n{traceback.format_exc()}")
        return {'status': False, 'msg': error_msg}


# =========================== #


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
    