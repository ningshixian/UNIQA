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

from contextlib import redirect_stdout, redirect_stderr
with open(os.devnull, 'w') as f, redirect_stdout(f), redirect_stderr(f):
    import jionlp
    import pandas as pd

from fastapi import FastAPI
from fastapi import APIRouter, HTTPException, Request, UploadFile
from fastapi import Depends, File, Body, File, Form, Query
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, validator
import uvicorn
import asyncio
# from pydantic import BaseModel, validator
# from starlette.requests import Request

from utils import logger, log_filter
from utils import data_cleaning
from configs.config import *
from responses import response_code
import application

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

faq_sys = application.get()
filters = application.get_filters()


# 知识检索使用(2025.7.24 修改)
class Item4cc(BaseModel):
    text: Union[str, List[str]]   # 原始/改写后的文本
    user_id: str='test'
    session_id: str='test'
    top_k: int=5  # 返回知识的个数，默认5
    search_strategy: str='hybrid'  # 检索策略（可选 bm25/embedding/hybrid）
    # query_extend: bool=False    # 是否进行查询扩展，默认 False
    # library_team: list[str]=['default']  # 检索库id
    exclude_team: list[int]=[]  # 排除检索哪些库，1: "FAQ知识库", 2: "一触即达意图", 4: "自定义寒暄", 5: "内置寒暄库",


@dataclass
class SearchConfig:
    """搜索配置类
    用于定义FAQ搜索的阈值和优先级配置

    匹配逻辑：
    - 大于高阈值→直出答案（精准匹配）
    - 小于低阈值→澄清话术（不回复）
    - 二者之间→出列表（模糊匹配）
    - 只有特定情况选择低优先级结果
    """
    high_threshold: float = 0.88
    low_threshold: float = 0.44  # # 0.6 0.44 0.35
    source_high_priority: List[int] = None
    source_low_priority: List[int] = None
    
    def __post_init__(self):
        if self.source_high_priority is None:
            self.source_high_priority = [1, 2]  # FAQ知识库, 一触即达意图
        if self.source_low_priority is None:
            self.source_low_priority = [4, 5]  # 自定义寒暄, 内置寒暄库


@dataclass
class SearchResult:
    """搜索结果数据类
    用于封装FAQ搜索的完整结果信息
    """
    text: str           # 处理后的查询文本
    origin_text: str    # 原始查询文本
    response4dm: str    # 传给DM用于前端展示的响应内容
    match_type: int     # 匹配类型（0：空 1：精准匹配 3：模糊匹配）
    confidence: float   # 置信度（top1结果的得分）
    threshold: Dict[str, float]
    detail_results: List[Dict[str, Any]]    # 详细的检索结果列表


class QueryProcessor:
    """文本处理类

    主要功能：
    1. 基础文本清洗（使用data_cleaning工具）
    2. 高级文本清洗（使用jionlp工具）
    """
    
    @staticmethod
    def standardize(query: str) -> str:
        if not query:
            return ""
            
        # # 基础文本清洗
        # query = data_cleaning.clean_text(query)
        
        # 高级文本清洗
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


class FAQSearchEngine:
    """FAQ搜索引擎主类
    
    主要功能：
    1. 输入验证和预处理
    2. 调用FAQ系统执行搜索
    3. 应用优先级逻辑筛选结果
    4. 处理意图和实体提取
    5. 构建最终响应
    """
    
    def __init__(self, faq_sys, config: SearchConfig = None):
        self.faq_sys = faq_sys
        self.config = config or SearchConfig()
        self.query_processor = QueryProcessor()
        
    def customized_search(self, 
               text: Union[str, List[str]],
               user_id: str = 'test',
               top_k: int = 5,
               search_strategy: str = 'hybrid',
               exclude_team: List[int] = None) -> SearchResult:
        """
        执行FAQ业务搜索逻辑
        
        Args:
            text: 输入文本
            user_id: 用户ID
            session_id: 会话ID
            top_k: 返回结果数
            search_strategy: 搜索策略
            exclude_team: 排除的库ID列表
            user_conditions: 用户条件
            
        Returns:
            SearchResult对象
        """
        # 初始化响应
        response = SearchResult(
            text=text,
            origin_text=text,
            response4dm="",     # 传给DM用于前端展示
            match_type=-1,      # top1回复类型 0：空 1：精准匹配 3：模糊匹配
            confidence=-1,      # top1 score
            threshold={"high": self.config.high_threshold, 
                      "low": self.config.low_threshold},    # 高/低阈值
            detail_results=[]   # faq检索的所有结果
        )

        # # 获取用户条件(需要调用外部接口)
        # car_type, ota_version = api(user_id)
        # user_conditions = {"car_type": car_type, "ota_version": ota_version}
        
        # 输入验证
        validation_error = self._validate_input(text, search_strategy, top_k)
        if validation_error:
            response.response4dm = validation_error
            return response
        
        # 文本预处理
        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list):
            response.response4dm = "输入text数据类型错误"
            return response
        # 数据清洗
        text = [self.query_processor.standardize(t) for t in text]
        
        if not text or not all(text):
            response.response4dm = "数据验证不通过-无效数据，请重新输入您的问题!"
            return response
        
        # # 车型标签2.0优化-0909：对 query 中提及的车型进行抽取，更新car_type参数
        # flag, entity_value, start_idx, end_idx = uie.entity_extract(text, "algorithmEntity", {})
        # if flag:
        #     car_type.append(entity_value)
        
        # 1、执行搜索
        search_results = self._execute_search(
            text, top_k, search_strategy, exclude_team
        )

        # # 2、复杂filters逻辑 → 筛选答案
        # save_idx_list = []
        # user_conditions = {"car_type": car_type, "ota_version": ota_version}
        # for doc in search_results:
        #     answers = doc.meta.get("answer_content_list", [])
        #     for i,ans in enumerate(answers):
        #         match_score = self.faq_sys._match_conditions(ans, user_conditions)  # 条件匹配度（0~1）
        #         if match_score >= 0.7:
        #             save_idx_list.append(i)
        #     answers = [answers[i] for i in save_idx_list]
        
        # 检索结果为空，直接返回
        if not search_results:
            response.response4dm = "未匹配到答案，检索结果为空！"
            return response
        
        # # 3、应用匹配优先级逻辑
        # search_results = self._apply_priority_logic(search_results)
        
        # 检索结果为空，直接返回
        if not search_results:
            response.response4dm = "未匹配到答案，检索结果为空！"
            return response
        
        # # 过滤掉低置信度结果(可选)
        # 保留top_k
        search_results = search_results[:top_k]
        
        # 5、构建响应
        response.text = text[0]
        response.response4dm = self._build_response4dm(search_results)
        response.match_type = self._classify_match_type(search_results[0]['score'])
        response.confidence = round(search_results[0]['score'], 4)
        response.detail_results = search_results
        
        return response
    
    def _validate_input(self, text: Any, search_strategy: str, 
                       top_k: int) -> Optional[str]:
        """验证输入参数"""
        if not text:
            return "输入不能为空"
        if search_strategy not in ['hybrid', 'bm25', 'embedding']:
            return "检索策略错误或暂不支持..."
        if not isinstance(top_k, int) or top_k <= 0:
            return "top_k必须是正整数"
        return None
    
    def _execute_search(self, text: List[str], top_k: int, 
                       search_strategy: str, exclude_team: List[int]) -> List[Dict]:
        """执行搜索并过滤结果"""
        # 执行搜索
        results = self.faq_sys.run(text, size=top_k, search_strategy=search_strategy)
        return results
    
    def _classify_match_type(self, score: float) -> int:
        """根据得分分类答案类型
        - 精准匹配：得分 >= 0.88
        - 模糊匹配：0.44 <= 得分 < 0.88
        - 不回复：得分 < 0.44
        """
        if score >= self.config.high_threshold:
            return 1  # 精准匹配
        elif score >= self.config.low_threshold:
            return 3  # 模糊匹配
        else:
            return 0  # 不回复
    
    def _build_response4dm(self, results: List[Dict]) -> str:
        """构建响应消息"""
        lines = [f"{x['score']:.4f}\t{x['content']}" for x in results]
        message = "\n".join(lines)
        if results:
            message += "\n\nTop1答案：" + results[0].get("content", "")
        return message


# 使用示例
def create_faq_engine(faq_sys):
    """创建FAQ搜索引擎实例"""
    config = SearchConfig(
        high_threshold=0.88,
        low_threshold=0.44,
        source_high_priority=[1, 2],
        source_low_priority=[4, 5]
    )
    
    engine = FAQSearchEngine(faq_sys, config)
    # engine.set_intent_extractor(intent_dict, param_dict, entity_dict)
    
    return engine


@router.get("/")
async def read_root():
    return response_code.resp_200(data={"message": "Welcome to the FAQ API"})


@router.post("/predict4cc", summary="FAQ检索式问答", name="FAQ检索")
@log_filter
def predict4cc(
    request: Request, 
    item: Item4cc
):
    """
    FAQ检索式问答接口
    
    接收用户查询请求，执行FAQ搜索并返回结果
    
    请求参数：
    - text: 用户查询文本
    - user_id: 用户ID
    - top_k: 返回结果数量
    - search_strategy: 搜索策略 ('hybrid', 'bm25', 'embedding')
    - exclude_team: 排除的数据源ID列表
    
    响应格式：
    - code: 状态码 (200表示成功，4001表示参数错误)
    - data: 搜索结果数据
    - message: 响应消息
    """
    text = item.text
    user_id = item.user_id
    top_k = item.top_k
    search_strategy = item.search_strategy
    exclude_team = item.exclude_team

    """FAQ检索式问答接口"""
    # 创建搜索引擎（可以在应用启动时创建并复用）
    engine = create_faq_engine(faq_sys)

    # 执行搜索
    response = engine.customized_search(
        text=text,
        user_id=user_id,
        top_k=top_k,
        search_strategy=search_strategy,
        exclude_team=exclude_team,
    )

    # 转换为响应格式
    response_data = response.__dict__
    
    if response.match_type == -1:  # 表示有错误
        return response_code.resp_4001(data=response_data)
    else:
        return response_code.resp_200(data=response_data)


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
