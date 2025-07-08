import socket
import time
import os
import subprocess


while True:
    time.sleep(20)  # 每隔20s检查一次服务
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)
    ip = "127.0.0.1"  # ip对应服务器的ip地址
    port = 8098
    result = sock.connect_ex((ip, port))  # 返回状态值
    if result == 0:
        pass
    else:
        print("Port %d is not open! Restarting..." % port)
        # run和popen最大的区别在于：run方法是阻塞调用，会一直等待命令执行完成或失败；popen是非阻塞调用，执行之后立刻返回，结果通过返回对象获取。
        subprocess.run("nohup gunicorn router:app -c configs/gunicorn_config_api.py > logs/router.log 2>&1 &", shell=True)
        # loader = subprocess.Popen(
        #     [
        #         "nohup",
        #         "gunicorn",
        #         "main:app",
        #         "-b", "0.0.0.0:9000",
        #         "-w", "1",
        #         "--threads", "100",
        #         "-k", "uvicorn.workers.UvicornWorker",
        #         "> logs/main.log 2>&1 &"
        #     ]
        # )
        # returncode = loader.wait()  # 阻塞直至子进程完成
        print("Service restarted.")
        # 在此处可以添加发送警报的代码

    sock.close()
