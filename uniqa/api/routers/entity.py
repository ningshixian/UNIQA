import os, sys
import json
from typing import Dict, List, Optional, Set, Tuple, Union, Any
from fastapi import APIRouter, Body, File, Form, HTTPException, Request, UploadFile
from fastapi.encoders import jsonable_encoder

# sys.path.append("..")
from uniqa.api.faq import FAQPipeline, DataPreprocessor
from uniqa import Document
from uniqa.logging import Loggers, logDog, log_filter
from uniqa.api.responses import response_code
from uniqa.api import application
from configs.config import *

from uniqa.utils.device import ComponentDevice
from uniqa.components.extractors import NamedEntityExtractor, NamedEntityExtractorBackend

router = APIRouter()


@router.post("/entity_extract", summary="算法实体抽取", name="算法实体抽取")
def entity_extract(
    request: Request, 
    text: str=Body(..., embed=True)
):
    """
    算法实体抽取
    """
    documents = [Document(content=text)]
    
    extractor = NamedEntityExtractor(
        backend=NamedEntityExtractorBackend.SPACY, 
        model="zh_core_web_trf", 
        device=ComponentDevice.from_str("mps"),
    )
    extractor.warm_up()
    results = extractor.run(documents=documents)["documents"]
    annotations = [NamedEntityExtractor.get_stored_annotations(doc) for doc in results]  # doc["named_entities"]
    print(annotations)

    match_entity = []
    if annotations:
        for i, annotation in enumerate(annotations):
            for item in annotation:
                start_idx = item['start']
                end_idx = item['end']
                entity = item['entity']
                
                match_entity.append({
                    'start': start_idx,  #实体位置
                    'end': end_idx,  #实体位置
                    'entity_name': entity, 
                    'entity_value': documents[i][start_idx:end_idx],  #实体值
                })
    # print(match_entity)
    return response_code.resp_200(data=match_entity)



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
