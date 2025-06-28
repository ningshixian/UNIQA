import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import traceback
from module.recall import Recall
from module.rank import PostRank
from configs.config import *
# from utils.log import logger, log_filter

"""
提供 FAQ 检索服务，包括recall 和 rank
"""


class FAQ(object):
    def __init__(self, qa_path_list):
        # logger.info('----------------- Initial -----------------')
        self.recall_module = Recall(qa_path_list)
        self.rank_module = PostRank(rerank_model_path)

    def search(self, query, size=5, search_strategy='hybrid'):
        # logger.info('----------------- FAQ start -----------------')
        # logger.info('query: {}'.format(query))

        if isinstance(query, str):
            query = [query]

        try:
            # 知识召回&排序
            recall_hits = self.recall_module.retrieve(query, topK=size, search_strategy=search_strategy)
            if search_strategy == "hybrid":
                results = self.rank_module.rerank(query, recall_hits[0], recall_hits[1], fusion=True)
            else:
                results = recall_hits
        except Exception as e:
            print(f"检索过程中发生错误：{e}")
            traceback.print_exc()
        
        # 返回第一个检索结果
        results = results[0]    # 取第一个query的检索结果
        # print("Top1答案：" + results[0]["answer"])
        # for x in results:
        #     print(f"{x['score']:.4f}\t{x['standard_sentence']}")

        # logger.info('----------------- FAQ end-----------------')
        return results


if __name__ == '__main__':
    
    qa_path_list=[qa4api_path, qa_qy_onetouch, qa_custom_greeting, qa_global_greeting]
    faq = FAQ(qa_path_list)
    faq.search('你好那个便携式充电器别的车子通用吗？', size=5, search_strategy='hybrid')

    while True:
        # 你好那个便携式充电器别的车子通用吗？
        query = input("Enter query: ")
        if query.lower() in ["exit", "quit"]:
            print("退出程序。")
            break
        try:
            faq.search(query, size=5, search_strategy='hybrid')
        except Exception as e:
            print(f"检索过程中发生错误：{e}")
