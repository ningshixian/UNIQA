
import sys
import time
import traceback
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import Log

"""
python主线程捕获子线程的方法:
https://www.jb51.net/article/142142.htm
https://www.it-swarm.net/zh/python/%E5%9C%A8python%E7%9A%84%E8%B0%83%E7%94%A8%E8%80%85%E7%BA%BF%E7%A8%8B%E4%B8%AD%E6%8D%95%E8%8E%B7%E7%BA%BF%E7%A8%8B%E7%9A%84%E5%BC%82%E5%B8%B8/970039453/

传入要调用的方法，并启用一个新的线程来运行这个方法
main方法中无法捕获子线程中的异常，原因在于start()方法将为子线程开辟一条新的栈，main方法的栈因此无法捕获到这一异常。

解决方法
1. 设置一个线程是否异常退出的的成员变量`flag` 
2. 当线程异常退出时，对`flag`作一标记，并通过sys.exc_info()和traceback回溯异常信息，简单地存储为线程对象的属性，直到调用join
3. 在join期间检查线程运行结束后`flag`标志位的值，重新引发它
4. 捕捉异常，记录错误日志
"""


# # 生成logger对象，全局唯一
# logger = Log._get_logger(log_to_file=True)


# 苏剑林大神代码
def parallel_apply(func, iterable, workers, max_queue_size, callback=None, dummy=False):
    """
    多进程或多线程地将func应用到iterable的每个元素中。
    注意这个apply是异步且无序的，也就是说依次输入a,b,c，但是
    输出可能是func(c), func(a), func(b)。
    参数：
        dummy: False是多进程/线性，True则是多线程/线性；
        callback: 处理单个输出的回调函数；
    """
    if dummy:
        from multiprocessing.dummy import Pool, Queue
    else:
        from multiprocessing import Pool, Queue

    in_queue, out_queue = Queue(max_queue_size), Queue()

    def worker_step(in_queue, out_queue):
        # 单步函数包装成循环执行
        while True:
            d = in_queue.get()
            r = func(d)
            out_queue.put(r)

    # 启动多进程/线程
    pool = Pool(workers, worker_step, (in_queue, out_queue))

    if callback is None:
        results = []

    # 后处理函数
    def process_out_queue():
        out_count = 0
        for _ in range(out_queue.qsize()):
            d = out_queue.get()
            out_count += 1
            if callback is None:
                results.append(d)
            else:
                callback(d)
        return out_count

    # 存入数据，取出结果
    in_count, out_count = 0, 0
    for d in iterable:
        in_count += 1
        while True:
            try:
                in_queue.put(d, block=False)
                break
            except six.moves.queue.Full:
                out_count += process_out_queue()
        if in_count % max_queue_size == 0:
            out_count += process_out_queue()

    while out_count != in_count:
        out_count += process_out_queue()

    pool.terminate()

    if callback is None:
        return results


# sents = ["a", "b"]
# results = parallel_apply(lambda x: print(x), sents, workers=5, max_queue_size=3, dummy=True)
# print(results)
# exit()


# 目前在用的并发方法
def thread_pool_apply(func, data_list):
    """线程池"""
    ss = 1000 * time.time()
    task_list = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        for i, q in enumerate(data_list):
            task = executor.submit(func, q)
            task_list.append(task)
        
        idx = 1
        for future in as_completed(task_list):
            resp = future.result()
            print(str(idx) + "-" + resp)
            idx += 1
            # result = resp.json()
            # assert resp.status_code == 200
            # assert result["msg"] == "succeed"
    print("并行平均耗时：", (1000*time.time() - ss)/len(data_list))   # 55ms


# def func(x):
#     return x+x
# data_list = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]
# thread_pool_apply(func, data_list)