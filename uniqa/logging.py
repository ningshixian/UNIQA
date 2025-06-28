import os
import sys
import time
import traceback
from functools import wraps
from typing import cast
from types import FrameType
import logging

from loguru import logger
from fastapi import status
from fastapi.responses import JSONResponse, Response
from fastapi.encoders import jsonable_encoder

"""
可以将 log 的配置和使用更加简单和方便
https://blog.csdn.net/qq_51967017/article/details/134045236

# # logger的完整配置
# logger.add(
#     sink=os.path.join(log_path, 'app.log'),
#     rotation='500 MB',                  # 日志文件最大限制500mb
#     retention='30 days',                # 最长保留30天
#     # format="{time}|{level}|{message}",  # 日志显示格式
#     compression="zip",                  # 压缩形式保存
#     encoding='utf-8',                   # 编码
#     level='INFO',                       # 日志级别
#     enqueue=True,                       # 默认是线程安全的，enqueue=True使得多进程安全
#     # backtrace=True,                   # 显示整个堆栈跟踪(包括变量值)来帮助您识别问题。应该在生产环境中将其设置为 False
#     # diagnose=True,                    # 显示整个堆栈跟踪(包括变量值)来帮助您识别问题。应该在生产环境中将其设置为 False
# )
"""


class Logger:
    def __init__(self):
        log_path = os.path.join('./uniqa/', 'logs')
        # 判断日志文件夹是否存在，不存则创建
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        
        # logger文件区分不同级别的日志
        log_file = os.path.join(log_path, f"{time.strftime('%Y-%m-%d').replace('-', '_')}.log")
        log_file_error = os.path.join(log_path, f"{time.strftime('%Y-%m-%d').replace('-', '_')}_error.log")

        self.logger = logger
        # 清空所有设置
        self.logger.remove()
        
        # 日志输出格式
        formatter = "{time:YYYY-MM-DD HH:mm:ss} | {level}: {message}"
        # 添加控制台输出的格式,sys.stdout为输出到屏幕;关于这些配置还需要自定义请移步官网查看相关参数说明
        self.logger.add(sys.stdout,
                        format="<green>{time:YYYYMMDD HH:mm:ss}</green> | "  # 颜色>时间
                               "{process.name} | "  # 进程名
                               "{thread.name} | "  # 进程名
                               "<cyan>{module}</cyan>.<cyan>{function}</cyan>"  # 模块名.方法名
                               ":<cyan>{line}</cyan> | "  # 行号
                               "<level>{level}</level>: "  # 等级
                               "<level>{message}</level>",  # 日志内容
                        )

        # 日志写入文件
        self.logger.add(log_file,  # 写入目录指定文件
                        format='{time:YYYYMMDD HH:mm:ss} - '  # 时间
                               "{process.name} | "  # 进程名
                               "{thread.name} | "  # 进程名
                               '{module}.{function}:{line} - {level} -{message}',  # 模块名.方法名:行号
                        encoding='utf-8',
                        retention='7 days',  # 设置历史保留时长
                        backtrace=True,  # 回溯
                        diagnose=True,  # 诊断
                        enqueue=True,  # 异步写入
                        rotation="00:00",  # 每日更新时间
                        # rotation="5kb",  # 切割，设置文件大小，rotation="12:00"，rotation="1 week"
                        # filter="my_module"  # 过滤模块
                        # compression="zip"   # 文件压缩
                        )

        # 日志写入文件
        self.logger.add(log_file_error,  # 写入目录指定文件
                        level="ERROR", 
                        format='{time:YYYYMMDD HH:mm:ss} - '  # 时间
                               "{process.name} | "  # 进程名
                               "{thread.name} | "  # 进程名
                               '{module}.{function}:{line} - {level} -{message}',  # 模块名.方法名:行号
                        encoding='utf-8',
                        retention='7 days',  # 设置历史保留时长
                        backtrace=True,  # 回溯
                        diagnose=True,  # 诊断
                        enqueue=True,  # 异步写入
                        rotation="00:00",  # 每日更新时间
                        # rotation="200 MB",  # 切割，设置文件大小, rotation="12:00"，rotation="1 week"
                        filter=lambda record: record["level"].name == "ERROR",   # 过滤模块
                        # compression="zip"   # 文件压缩
                        )

    def init_config(self):
        LOGGER_NAMES = ("uvicorn.asgi", "uvicorn.access", "uvicorn")
 
        # change handler for default uvicorn logger
        logging.getLogger().handlers = [InterceptHandler()]
        for logger_name in LOGGER_NAMES:
            logging_logger = logging.getLogger(logger_name)
            logging_logger.handlers = [InterceptHandler()]
 
    def get_logger(self):
        return self.logger


class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = str(record.levelno)
 
        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:  # noqa: WPS609
            frame = cast(FrameType, frame.f_back)
            depth += 1
 
        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage(),
        )


"""logger 使用方式2
装饰器来捕获代码异常&记录日志
"""

Loggers = Logger()
logDog = Loggers.get_logger()

def log_filter(func):
    """装饰器来捕获代码异常&记录日志"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = 1000 * time.time()
        logDog.info(f"=============  Begin: {func.__name__}  =============")
        logDog.info(f"Request: {kwargs['request'].url}")
        logDog.info(f"Args: {kwargs['item2'] if 'item2' in kwargs else kwargs['item']}")
        try:
            rsp = func(*args, **kwargs)
            logDog.info(f"Response: {rsp.body.decode('utf-8')}") 
            end = 1000 * time.time()
            logDog.info(f"Time consuming: {end - start}ms")
            logDog.info(f"=============   End: {func.__name__}   =============\n")
            return rsp
        except Exception as e:
            logDog.error(traceback.format_exc())  # 错误日志 repr(e)
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=jsonable_encoder({
                    'code': 500,
                    'message': "Internal Server Error"
                })
            )
    return wrapper


@log_filter
def main():
    print("ceshi")


__all__ = ["logDog", "log_filter"]
