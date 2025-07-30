import json
from typing import Dict, List, Optional, Set, Tuple, Union, Any
from fastapi import APIRouter, Body, File, Form, HTTPException, Request, UploadFile
from fastapi.encoders import jsonable_encoder

from uniqa.api.faq import FAQPipeline, DataPreprocessor
from uniqa import Document
from uniqa.logging import Loggers, logDog, log_filter
from uniqa.api.responses import response_code
from uniqa.api import application
from configs.config import *

router = APIRouter()
faq_sys = application.get()


@router.post("/similarity", summary="相似度计算", name="相似度计算")
def similarity(
    request: Request, 
    text1: str=Body(..., embed=True),
    text2: str=Body(..., embed=True)
):
    """
    计算两个文本之间的向量相似度
    """
    if not text1 or not text2:
        return response_code.resp_4001(data="输入不能为空")
    vec1 = faq_sys.text_embedder.run(text1)["embedding"]
    vec2 = faq_sys.text_embedder.run(text2)["embedding"]

    # 计算并返回点积和余弦相似度
    from sklearn.metrics.pairwise import cosine_similarity
    score = cosine_similarity(vec1, vec2)
    # 要保证接口的输出是字典格式，且避免 float 类型数据
    return response_code.resp_200(data={'cosine_similarity': str(score[0])})

