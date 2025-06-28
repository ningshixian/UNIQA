#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/1/29 21:11
# @Author  : CoderCharm
# @File    : sys_api.py
# @Software: PyCharm
# @Github  : github/CoderCharm
# @Email   : wg_python@163.com
# @Desc    :
"""
实体抽取接口 需要验证的条件
"""
from pydantic import BaseModel, validator
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union


# 请求参数参考 https://www.alibabacloud.com/help/zh/open-search/llm-intelligent-q-a-version/search-knowledge?spm=a2c63.p38356.0.0.75fc6cb9BDr4Tj
class Item(BaseModel):
    text: Union[str, List[str]]
    session_id: str='test'
    top_n: int=5  # 返回知识的个数
    search_strategy: str='hybrid'  # 检索策略（可选 bm25/embedding/hybrid）


# 知识推荐使用
class Item4rec(BaseModel):
    text: Union[str, List[str]]
    ticketCode: str     # 工单编码
    carType: str="与车型无关"
    topN: int=3  # 返回知识的个数
    message: Optional[List[Dict]]
    search_strategy: str='hybrid'  # 检索策略（可选 bm25/embedding/hybrid）
    

# 知识检索使用
class Item4cc(BaseModel):
    text: Union[str, List[str]]   # 原始/改写后的文本
    user_id: str='test'
    session_id: str='test'
    top_n: int=5  # 返回知识的个数，默认5
    search_strategy: str='hybrid'  # 检索策略（可选 bm25/embedding/hybrid）
    query_extend: bool=False    # 是否进行查询扩展，默认 False
    # library_team: list[str]=['default']  # 检索库id
    exclude_team: list[int]=[]  # 排除检索哪些库，1: "FAQ知识库", 2: "一触即达意图", 4: "自定义寒暄", 5: "内置寒暄库",
    car_type: list[str]=[]  # 车辆类型


class Item4sim(BaseModel):
    text1: str
    text2: str


class Item4emb(BaseModel):
    text: Union[str, List[str]]


class Item4ner(BaseModel):
    text: str
    entity_name: str
    param_name: str
