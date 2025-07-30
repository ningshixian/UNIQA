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
loguru 可以将log的配置和使用更加简单和方便
https://blog.csdn.net/qq_51967017/article/details/134045236

# 彩色格式
from colorlog import ColoredFormatter
formatter = ColoredFormatter(
    "%(asctime)s - %(log_color)s%(levelname)-8s%(reset)s - %(blue)s%(message)s",
    datefmt='%Y-%m-%d %H:%M:%S',
    reset=True,
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red',
    },
    secondary_log_colors={},
    style='%'
)
# --- 日志配置 ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False  # 防止日志传播到根logger
# 控制台输出配置
sh = logging.StreamHandler(sys.stdout)  #往屏幕上输出
sh.setFormatter(formatter) #设置屏幕上显示的格式
# 添加 handler 到 logger
logger.addHandler(sh)
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
        fmt = ("<green>{time:YYYYMMDD HH:mm:ss}</green> | "  # 颜色>时间
            "{process.name} | "  # 进程名
            "{thread.name} | "  # 进程名
            "<cyan>{module}</cyan>.<cyan>{function}</cyan>"  # 模块名.方法名
            ":<cyan>{line}</cyan> | "  # 行号
            "<level>{level}</level>: "  # 等级
            "<level>{message}</level>"
        )
        # 添加控制台输出的格式,sys.stdout为输出到屏幕;关于这些配置还需要自定义请移步官网查看相关参数说明
        self.logger.add(sys.stdout, format=fmt)

        # 日志写入文件
        fmt = "{time:YYYY/MM/DD at HH:mm:ss} | {level} | {file}:{line} | {message}"
        self.logger.add(log_file,  # 写入目录指定文件
                        format=fmt, 
                        encoding='utf-8',
                        retention='7 days',  # 设置历史保留时长
                        backtrace=True,  # 回溯
                        diagnose=True,  # 诊断
                        enqueue=True,  # 异步写入
                        rotation="00:00",  # 每日更新时间 "12:00"，"1 week"
                        # rotation="200 MB",  # 切割，设置文件大小
                        # filter=lambda record: record["level"].name == "INFO",  # 过滤模块
                        # compression="zip",   # 文件压缩
                        # serialize=True,  # Loguru 会将全部日志消息转换为 JSON 格式保存
                        )

        # 日志写入文件
        self.logger.add(log_file_error,  # 写入目录指定文件
                        level="ERROR", 
                        format=fmt, 
                        encoding='utf-8',
                        retention='7 days',  # 设置历史保留时长
                        backtrace=True,  # 回溯
                        diagnose=True,  # 诊断
                        enqueue=True,  # 异步写入
                        rotation="00:00",  # 每日更新时间
                        # rotation="200 MB",  # 切割，设置文件大小, rotation="12:00"，rotation="1 week"
                        filter=lambda record: record["level"].name == "ERROR",   # 过滤模块
                        # compression="zip",   # 文件压缩
                        # serialize=True,  # Loguru 会将全部日志消息转换为 JSON 格式保存
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
