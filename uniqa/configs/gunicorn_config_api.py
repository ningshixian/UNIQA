# gunicorn -c config/gunicorn_config.py main:app
import multiprocessing
import uvicorn

# ip + port
bind = "0.0.0.0:8098"

# worker超时时间，默认是30秒，超时重启
# 对于较重的任务，需要设置一个较大的超时时间，避免频繁WORKER TIMEOUT，导致无限重启
timeout = 240   # 120s已经不够用了

# 并行工作进程数
# workers = multiprocessing.cpu_count() * 2 + 1  # 内存占用大
workers = 1

# # 每个进程开启的线程数(此配置只适用于gthread 进程工作方式)
# # 最大的并发请求数就是 worker * 线程
# threads = 200

# 监听队列的最大连接数 (建议 64-2048)
backlog = 1024

# 工作模式：sync(同步), eventlet(并发), gevent(协程)
worker_class = "uvicorn.workers.UvicornWorker"

# # 连接的存活时间/等待时间，通常设置在1-5秒范围内。
# # 如果在存活时间内未收到新数据，则关闭保持活动状态的连接。
# keep_alive = 30

# 设置守护进程,False将进程交给supervisor管理,True是后台运行
daemon = False

# # 设置日志记录水平
# loglevel = 'info'

# # 设置进程文件目录
# pidfile = 'logs/gunicorn.pid'
# # 设置访问日志和错误信息日志路径
# accesslog = 'logs/gun-access.log'
# errorlog = 'logs/gun-error.log'

# # 在代码改变时自动重启
# reload=true

# # 预加载应用，启动时加载应用，避免每次请求都加载应用，提升性能，但是在加载向量时报错！
# preload_app=True 