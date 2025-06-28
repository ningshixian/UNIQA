import sys
import copy
from collections import defaultdict
from typing import List, Union, Dict, Any, Tuple
from numpy import array, ndarray
import traceback

import torch
# from sentence_transformers import SentenceTransformer
# from sentence_transformers import CrossEncoder
from rerankers import Reranker
from rerankers import Document

from pathlib import Path
# 获取当前文件的绝对路径
current_dir = Path(__file__).resolve().parent
# 获取上一级目录的路径
parent_parent_dir = current_dir.parent
# 将上一级目录添加到 sys.path
sys.path.append(str(parent_parent_dir))
from configs.config import *

"""https://github.com/AnswerDotAI/rerankers
"""

class PostRank(object):
    def __init__(self, rerank_model_path):
        # Cross-encoder default. You can specify a 'lang' parameter to load a multilingual version!
        self.ranker = Reranker(
            rerank_model_path,
            model_type='cross-encoder', 
            device="cuda",
            # dtype=torch.float16, 
            # batch_size=64, 
            # lang="zh", 
        )

    def rerank(
        self,
        queries: Union[List[str], ndarray],
        merge_hits: List,
        semantic_hits: List,
        fusion: bool=True,
    ):
        semantic_hits_copy = copy.deepcopy(semantic_hits)   # rerank会改变该变量内容?
        merge_hits_copy = copy.deepcopy(merge_hits)   # rerank会改变该变量内容?

        # cross-encoder重排
        results = [
            self.rerank4query(query, docs)
            for query, docs in zip(queries, merge_hits_copy)
        ]

        if fusion and semantic_hits:
            # 线性加权融合
            results = [
                self.weighted_fusion(semantic_hit, rank_hit, weights=[0.5, 0.5])
                for semantic_hit, rank_hit in zip(semantic_hits_copy, results)
            ]
        return results

    def rerank4query(self, query, docs):
        """
        使用reranker对【多路召回文档】进行重排。
        参数:
            docs: 一个包含文档的列表，每个文档是一个字典，至少包含'standard_sentence', 'match_sentence', 'answer'键。
            query: 一个字符串，代表查询。
        返回:
            一个根据reranker分数重新排序的文档列表。
        """
        # 确保输入的docs和query符合预期的数据结构
        if not isinstance(docs, list) or not isinstance(query, str):
            raise ValueError("docs必须是列表，query必须是字符串。")
        if len(docs)==1:    # query="什么时候发货", docs=['什么时候发货']
            return docs

        # 构建 cross 模型输入
        _docs = [
            Document(text=doc["match_sentence"], doc_id=did)    # , metadata=doc
            for did, doc in enumerate(docs)
        ]
        results = self.ranker.rank(query=query, docs=_docs)

        # print(results.top_k(topK))
        # print(results.top_k(topK)[0].text) 
        # results.get_score_by_docid(1)

        # results = self.ranker.rank(
        #     query=query, 
        #     docs=[doc["match_sentence"] for doc in docs], 
        #     doc_ids=list(range(len(docs)))
        # )
        
        # 得分归一化 - sigmoid
        new_docs = []
        score_logits = [result.score for result in results]
        scores = torch.sigmoid(torch.tensor(score_logits)).tolist()  # 0~1 
        for ix, res in enumerate(results):
            docs[res.doc_id]["score"] = scores[ix]
            new_docs.append(docs[res.doc_id])

        new_docs.sort(key=lambda x: x["score"], reverse=True)
        return new_docs

    def weighted_fusion(self, hits1, hits2, weights=[0.9, 0.1]):
        """
        对召回结果进行加权融合
        """
        combined_results = defaultdict(lambda: {"score": 0})
        # 对语义召回结果进行加权
        for hit in hits1:
            hit["score"] *= weights[0]
            combined_results[hit["standard_sentence"]] = hit
        # 对BM25召回结果进行加权，并与语义召回结果融合
        for hit2 in hits2:
            hit2['score'] *= weights[1]
            sen_hit2 = hit2['standard_sentence']
            if sen_hit2 in combined_results:
                combined_results[sen_hit2]['score'] += hit2['score']
            else:
                combined_results[sen_hit2] = hit2

        # 按分数降序排序并返回结果
        combined_results = sorted(
            combined_results.items(), key=lambda x: x[1]["score"], reverse=True
        )
        return list(map(lambda x: x[1], combined_results))


if __name__ == "__main__":

    from recall import Recall
    qa_path_list=[qa4api_path, qa_qy_onetouch, qa_custom_greeting, qa_global_greeting]
    recall_module = Recall(qa_path_list)
    rank_module = PostRank(rerank_model_path)

    topK = 5
    while True:
        # 你好那个便携式充电器别的车子通用吗？
        query = input("Enter query: ")
        if query.lower() in ["exit", "quit"]:
            print("退出程序。")
            break

        try:
            merge_hits, semantic_hits = recall_module.retrieve(query, topK=topK, search_strategy='hybrid')
            print(merge_hits[0][:topK])
            results = rank_module.rerank([query], merge_hits, semantic_hits, fusion=True)
            print(results[0][:topK])
            results = rank_module.rerank([query], merge_hits, semantic_hits, fusion=False)
            print(results[0][:topK])

        except Exception as e:
            print(f"检索过程中发生错误：{e}")
            traceback.print_exc()
            continue  # 出现异常时，继续下一轮循环

        # for x in results:
        #     print(f"{x['score']:.4f}\t{x['standard_sentence']}")
        # print("Top1答案：" + results[0]["answer"])
