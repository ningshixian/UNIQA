import os
import sys
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.getcwd())
sys.path.append(os.path.abspath('.'))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import status
from fastapi.responses import JSONResponse, Response
from fastapi.responses import StreamingResponse
from fastapi import APIRouter, Depends, File, UploadFile, Query

import re
from datetime import datetime
import importlib
import traceback
from pydantic import BaseModel, validator
from starlette.requests import Request
from starlette.testclient import TestClient
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

# 添加CORS中间件（允许所有来源测试）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,  # 允许携带凭证
    allow_methods=["*"],  # 允许所有HTTP方法
    allow_headers=["*"],  # 允许所有请求头
)

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


# @app.post("/rag-stream")
# async def rag_stream(request: Request):
#     data = await request.json()
#     query = data.get("query", "")

#     from uniqa.api.rag_async import generate_stream
#     # 创建一个 StreamingResponse 实例，用于处理流式响应
#     return StreamingResponse(
#         generate_stream(query),     # 流式生成器
#         media_type="text/event-stream",
#         headers={
#             "Cache-Control": "no-cache",
#             "X-Accel-Buffering": "no"
#         }
#     )


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
        # # 获取 Milvus 中所有文档的 id 和 meta
        # # 注意：对于非常大的数据集，一次性获取所有文档可能效率不高。
        # # 更好的方法是分批获取或使用更高级的同步策略（见后文）。
        # # 这里为了演示，我们获取全部。
        # milvus_docs = faq.milvus_document_store.filter_documents()
        # milvus_data_map = {doc.id: doc.meta for doc in milvus_docs}

        # --- 3. 更新和添加 (UPDATE & ADD) ---
        print("\n--- 3. 更新和添加 (UPDATE & ADD) ---")

        # 模拟数据更新：答案内容变了，并且增加了一个新的相似问题
        # update_and_add_docs = preprocessor.load_data("uniqa/data/update_robot_know_sim.json")     # 2
        update_and_add_docs = preprocessor.load_data("uniqa/data/update_robot_know.json")           # 1
        print(f"从更新后的 JSON 生成了 {len(update_and_add_docs)} 个 Haystack Document:")
        for doc in update_and_add_docs:
            print(f"  - ID: {doc.id}, Content: '{doc.content}'")

        # 嵌入并使用 OVERWRITE 策略写入
        from uniqa.document_stores.types import DuplicatePolicy
        embedded_update_docs = faq.doc_embedder.run(documents=update_and_add_docs)["documents"]
        # old: MilvusDocumentStore.write_documents
        # new: MilvusDocumentStore.update_documents
        count = faq.milvus_document_store.update_documents(
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


if __name__ == "__main__":
    import uvicorn
    config = uvicorn.Config(
        "api:app", 
        host='0.0.0.0', 
        port=8091, 
        workers=1, 
        # limit_concurrency=200,  # 最大并发数，默认 None
        # reload=True,
        # debug=True,
    )
    server = uvicorn.Server(config)
    # 将uvicorn输出的全部让loguru管理
    Loggers.init_config()
    server.run()
    