import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import warnings
warnings.filterwarnings("ignore")

import re
import time
from datetime import datetime
from dataclasses import dataclass
import importlib
import traceback
from collections import OrderedDict
from typing import Dict, List, Optional, Set, Tuple, Union, Any
import json
import requests
import torch
import gc
import logging
import pandas as pd

from fastapi import FastAPI
from fastapi import APIRouter, HTTPException, Request, UploadFile
from fastapi import Depends, File, Body, File, Form, Query
from fastapi.responses import JSONResponse, Response
import uvicorn
import asyncio
# from pydantic import BaseModel, validator
# from starlette.requests import Request

from uniqa.api.faq import FAQPipeline, DataPreprocessor
from uniqa import Document
from uniqa.logging import Loggers, logDog, log_filter
from uniqa.api.responses import response_code
from uniqa.api import application
from configs.config import *

"""
# Uvicorn 为单进程的 ASGI server ，而 Gunicorn 是管理运行多个 Uvicorn ，以达到并发与并行的最好效果。
CUDA_VISIBLE_DEVICES=3 python api.py 
gunicorn api:router --bind=0.0.0.0:8091 --workers=1 -k uvicorn.workers.UvicornWorker
nohup gunicorn api:router -c configs/gunicorn_config_api.py > logs/api.log 2>&1 &
"""

router = APIRouter()
# router = FastAPI(
#     title="CallCenter Knowledge Semantic Search",
#     description="基于售后知识平台，进行语义检索，召回相似问题",
# )

# 全局变量声明
global_u_docs: List[Dict] = []
faq_sys = application.get()


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


@router.get("/full_update")
def full_update(request: Request):
    """
    更新全局变量、重新加载Faiss和BM25索引
    重新初始化了模型，虽然最后任务结束，但是并不会释放显存，最终会显存溢出，考虑 del+gc
    https://wjwsm.top/2023/06/11/NLP%E6%A8%A1%E5%9E%8B%E9%83%A8%E7%BD%B2%E6%80%BB%E7%BB%93(fastapi+uvicorn)/
    """
    try:
        logDog.info("开始全量更新FAQ系统...")

        global faq_sys, global_u_docs
        global_u_docs = []

        preprocessor = DataPreprocessor()
        docs = preprocessor.load_data(data_path="uniqa/data/demo.json")
        faq = FAQPipeline(is_whitening=False)
        faq.setup_milvusDB_retriever(docs, application.get_filters())

        torch_gc()  # 随着更新次数增多，显存占用会变大，所以顶一个 torch_gc() 方法完成对显存的回收

        logDog.info("FAQ系统全量更新成功\n")
        return {'status': True, 'msg': 'Successfully updated all FAQ components'}
    except Exception as e:
        error_msg = f"【full_update】接口更新失败！发生错误: {str(e)}"
        logDog.error(f"{error_msg}\n{traceback.format_exc()}")
        return {'status': False, 'msg': error_msg}


@router.get("/incremental_update")
def incremental_update(request: Request):
    """增量更新FAQ知识库 
    统一通过 write_documents 和 delete_documents 实现增删改，而无需每次都重新索引所有数据。
    """
    try:
        logDog.info(f"开始 知识增量更新")
        
        # # 获取 Milvus 中所有文档的 id 和 meta
        # # 注意：对于非常大的数据集，一次性获取所有文档可能效率不高。
        # # 更好的方法是分批获取或使用更高级的同步策略（见后文）。
        # # 这里为了演示，我们获取全部。
        # milvus_docs = faq.milvus_document_store.filter_documents()
        # milvus_data_map = {doc.id: doc.meta for doc in milvus_docs}

        # --- 3. 更新和添加 (UPDATE & ADD) ---
        print("\n--- 3. 更新和添加 (UPDATE & ADD) ---")

        # 模拟数据更新：答案内容变了，并且增加了一个新的相似问题
        preprocessor = DataPreprocessor()
        update_and_add_docs = preprocessor.load_data("uniqa/data/update_robot_know.json")           # 1
        print(f"从更新后的 JSON 生成了 {len(update_and_add_docs)} 个 Haystack Document:")
        for doc in update_and_add_docs:
            print(f"  - ID: {doc.id}, Content: '{doc.content}'")

        # 嵌入并使用 OVERWRITE 策略写入
        from uniqa.document_stores.types import DuplicatePolicy
        embedded_update_docs = application.get().doc_embedder.run(documents=update_and_add_docs)["documents"]
        # old: MilvusDocumentStore.write_documents
        # new: MilvusDocumentStore.update_documents
        count = application.get().milvus_document_store.update_documents(
            embedded_update_docs,
            policy=DuplicatePolicy.OVERWRITE
        )

        # 更新全局变量
        global global_u_docs
        global_u_docs.extend(embedded_update_docs)

        torch_gc()  # 随着更新次数增多，显存占用会变大，所以顶一个 torch_gc() 方法完成对显存的回收

        logDog.info(f"know 知识增量更新成功\n")
        logDog.info(f"\n增量更新操作写入了 {count} 份文档。")
        logDog.info(f"当前 DocumentStore 中的文档总数: {application.get().milvus_document_store.count_documents()}")
        return {'status': True, 'msg': f'Successfully updated {len(update_and_add_docs)} entries'}
    except Exception as e:
        error_msg = f"【incremental_update】 增量更新索引时发生错误: {str(e)}"
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
