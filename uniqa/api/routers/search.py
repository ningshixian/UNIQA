import json
from typing import Dict, List, Optional, Set, Tuple, Union, Any
from fastapi import APIRouter, Body, File, Form, HTTPException, Request, UploadFile
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, validator

from uniqa.api.faq import FAQPipeline, DataPreprocessor
from uniqa import Document
from uniqa.logging import Loggers, logDog, log_filter
from uniqa.api.responses import response_code
from uniqa.api import application
from configs.config import *

router = APIRouter()


@router.post("/search")
def search(
    request: Request, 
    query: Union[str, List[str]]= Body(..., embed=True),
    session_id: str= Body(default='test', embed=True),
    top_k: int= Body(default=5, embed=True),  # 返回知识的个数
    search_strategy: str= Body(default='hybrid', embed=True),  # 检索策略（可选 sparse/embedding/hybrid）
    car_type: list[str]= Body(default=None, embed=True),
    ota_version: list[str]= Body(default=None, embed=True),
):
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

    # # 数据预处理
    # query = list(map(_text_standardize, query))

    # 第一步：检索召回相关文档
    results = application.get().run(query, top_k=top_k, search_strategy=search_strategy)
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
            match_score = application.get()._match_conditions(ans, user_conditions)  # 条件匹配度（0~1）
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