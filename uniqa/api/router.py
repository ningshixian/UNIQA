import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from fastapi import APIRouter, Depends
from fastapi import FastAPI, Request, Response
import traceback
import uvicorn

from api import router as faq_router
from api4rec import router as rec_router

"""
APIRouter 是一个组织代码的方式，它允许你将应用程序分割成组件，每个组件都有自己的路由集合。

# 推荐启动方式 main指当前文件名字 app指FastAPI实例化后对象名称
uvicorn router:app --host=0.0.0.0 --port=8091 --workers=1

# Uvicorn 为单进程的 ASGI server ，而 Gunicorn 是管理运行多个 Uvicorn ，以达到并发与并行的最好效果。
gunicorn router:app -b 0.0.0.0:8098 -w 1 -t 50 -k uvicorn.workers.UvicornWorker
nohup gunicorn router:app -c configs/gunicorn_config_api.py > logs/router.log 2>&1 &
"""

api_router = APIRouter()
api_router.include_router(faq_router)
api_router.include_router(rec_router)

# api_v1_router.include_router(auth_router, prefix="/admin/auth", tags=["用户"])
# # api_v1_router.include_router(items_router, tags=["测试API"], dependencies=[Depends(check_jwt_token)])
# # check_authority 权限验证内部包含了 token 验证 如果不校验权限可直接 dependencies=[Depends(check_jwt_token)]
# api_v1_router.include_router(items_router, tags=["测试接口"], dependencies=[Depends(check_authority)])
# api_v1_router.include_router(scheduler_router, tags=["任务调度"],  dependencies=[Depends(check_authority)])
# api_v1_router.include_router(sys_api_router, tags=["服务API管理"],  dependencies=[Depends(check_authority)])
# api_v1_router.include_router(sys_casbin_router, tags=["权限API管理"],  dependencies=[Depends(check_authority)])


def create_app() -> FastAPI:
    """
    将FatAPI核心对象包装成函数返回，然后在主目录main.py调用启动
    :return:
    """
    app = FastAPI(
        title="CallCenter Knowledge Semantic Search",
        description="基于售后知识平台，进行语义检索，召回相似问题",
    )

    # 其余的一些全局配置可以写在这里 多了可以考虑拆分到其他文件夹

    # # 跨域设置
    # register_cors(app)

    # 注册路由
    register_router(app)

    # # 注册捕获全局异常
    # register_exception(app)

    # # 请求拦截
    # register_hook(app)

    # # 取消挂载在 request对象上面的操作，感觉特别麻烦，直接使用全局的
    # register_init(app)

    return app


def register_router(app: FastAPI) -> None:
    """
    注册路由
    :param app:
    :return:
    """
    # 项目API
    app.include_router(
        api_router,
    )  # 注册


app = create_app()


if __name__ == "__main__":
    # 输出所有的路由
    for route in app.routes:
        if hasattr(route, "methods"):
            print({"path": route.path, "name": route.name, "methods": route.methods})

    uvicorn.run(
        app="router:app", 
        host="0.0.0.0", 
        port=8098, 
        workers=1,  # 工作进程数
        # backlog=2048,   # 等待处理的最大连接数。默认为2048
        # limit_concurrency=200,  # 最大并发数，默认 None
        # reload=True,
        # debug=True,
    )
