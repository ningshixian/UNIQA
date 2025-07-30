"""
FastAPI application module
"""
import os
import sys
import inspect
from fastapi import APIRouter, Depends, FastAPI
# from httpx import AsyncClient
# from fastapi_mcp import FastApiMCP
# # pip install fastapi-mcp  # need Python 3.10+

from uniqa.api.faq import FAQPipeline, DataPreprocessor
from uniqa import Document
from configs.config import *

"""
uvicorn "application:app" --host 0.0.0.0 --port 8092 --workers 1 --reload
nohup gunicorn application:app -c configs/gunicorn_config_api.py > logs/app.log 2>&1 &
"""


# 提供全局访问点
# 重要组件都可以作为全局单例访问，避免了重复初始化的开销
def get():
    """
    Returns global API instance.

    Returns:
        API instance
    """

    return INSTANCE

def get_filters():
    return filters


def create() -> FastAPI:
    """
    Creates a FastAPI instance.
    """

    # Application dependencies
    dependencies = []

    # # Default implementation of token authorization
    # token = os.environ.get("TOKEN")
    # if token:
    #     dependencies.append(Depends(Authorization(token)))

    # # Add custom dependencies
    # deps = os.environ.get("DEPENDENCIES")
    # if deps:
    #     for dep in deps.split(","):
    #         # Create and add dependency
    #         dep = APIFactory.get(dep.strip())()
    #         dependencies.append(Depends(dep))

    # Create FastAPI application
    return FastAPI(
        title="FAQ API",
        description="语义检索，召回相似问题",
        lifespan=lifespan, 
        dependencies=dependencies if dependencies else None
    )


def apirouters():
    """
    输出所有available路由
    """

    # # Get handle to api module
    # api = sys.modules[".".join(__name__.split(".")[:-1])]
    # # api = sys.modules["faq-semantic-retrieval.routers"]

    # 尝试导入指定模块
    import importlib
    api = importlib.import_module("routers")

    available = {}
    for name, rclass in inspect.getmembers(api, inspect.ismodule):
        if hasattr(rclass, "router") and isinstance(rclass.router, APIRouter):
            available[name.lower()] = rclass.router

    # print("routers: ", available)
    return available


def lifespan(application):
    """
    FastAPI lifespan event handler.

    Args:
        application: FastAPI application to initialize
    """

    # pylint: disable=W0603
    global filters, faq, INSTANCE


    # Load YAML settings (这个配置可以用于控制注册那些路由！)
    config = {
        "embeddings": None, 
        "entity": None, 
        "similarity": None, 
        "faq_search_engine": None, 
        "faq_recommend_engine": None, 
        "update_api": None, 
        "rag": None, 
        "": None, 
    }

    # # Instantiate API instance
    # api = os.environ.get("API_CLASS")
    # # api = "txtai.api.API"
    # INSTANCE = APIFactory.create(config, api) if api else API(config)
    
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
    INSTANCE = faq

    # Get all known routers
    routers = apirouters()

    # Conditionally add routes based on configuration
    for name, router in routers.items():
        if name in config:
            application.include_router(router)  # 注册路由

    # # Add Model Context Protocol (MCP) service, if applicable
    # if config.get("mcp"):
    #     mcp = FastApiMCP(application, http_client=AsyncClient(timeout=100))
    #     mcp.mount()

    yield


def start():
    """
    Runs application lifespan handler.
    """

    list(lifespan(app))


# FastAPI instance txtai API instances
app, filters, faq, INSTANCE = create(), None, None, None


if __name__ == "__main__":
    # 输出所有的路由
    for route in app.routes:
        if hasattr(route, "methods"):
            print({"path": route.path, "name": route.name, "methods": route.methods})

    import uvicorn
    uvicorn.run(
        app="application:app", 
        host="0.0.0.0", 
        port=8098, 
        workers=1,  # 工作进程数
        # backlog=2048,   # 等待处理的最大连接数。默认为2048
        # limit_concurrency=200,  # 最大并发数，默认 None
        # reload=True,
        # debug=True,
    )