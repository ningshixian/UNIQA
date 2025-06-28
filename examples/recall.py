import sys
from collections import defaultdict
from typing import List, Union, Dict, Any, Tuple
from numpy import array, ndarray
import traceback

from pathlib import Path
# 获取当前文件的绝对路径
current_dir = Path(__file__).resolve().parent
# 获取上一级目录的路径
parent_parent_dir = current_dir.parent
# 将上一级目录添加到 sys.path
sys.path.append(str(parent_parent_dir))
from configs.config import *
from uniqa.components.indexs.faiss_index import FaissSearcher

# from training.bm25.bm25_sparse import BM25Model
import os
from typing import Union, List
from collections import OrderedDict
import jieba
from bm25s.tokenization import Tokenized
# from rank_bm25 import BM25Okapi
import bm25s
from tqdm.auto import tqdm
import numpy as np
import pandas as pd


def tokenize(
    texts,
    return_ids: bool = True,
    show_progress: bool = False,
    leave: bool = False,
) -> Union[List[List[str]], Tokenized]:
    if isinstance(texts, str):
        texts = [texts]

    corpus_ids = []
    token_to_index = {}

    for text in tqdm(
        texts, desc="Split strings", leave=leave, disable=not show_progress
    ):

        splitted = jieba.lcut(text, HMM=False)
        doc_ids = []

        for token in splitted:
            if token not in token_to_index:
                token_to_index[token] = len(token_to_index)

            token_id = token_to_index[token]
            doc_ids.append(token_id)

        corpus_ids.append(doc_ids)

    # Create a list of unique tokens that we will use to create the vocabulary
    unique_tokens = list(token_to_index.keys())

    vocab_dict = token_to_index

    # Return the tokenized IDs and the vocab dictionary or the tokenized strings
    if return_ids:
        return Tokenized(ids=corpus_ids, vocab=vocab_dict)

    else:
        # We need a reverse dictionary to convert the token IDs back to tokens
        reverse_dict = unique_tokens
        # We convert the token IDs back to tokens in-place
        for i, token_ids in enumerate(
            tqdm(
                corpus_ids,
                desc="Reconstructing token strings",
                leave=leave,
                disable=not show_progress,
            )
        ):
            corpus_ids[i] = [reverse_dict[token_id] for token_id in token_ids]

        return corpus_ids

bm25s.tokenize = tokenize


def min_max_normalization(scores):
    min_value = np.min(scores)
    max_value = np.max(scores)
    if min_value == max_value:
        # raise ValueError("Minimum and maximum values are the same, cannot normalize.")
        return scores
    # 应用最小-最大标准化公式
    normalized_data = (scores - min_value) / (max_value - min_value)
    return normalized_data


def normalization10(scores):
    # 将 BM25 分数除以本身加上 10
    normalized_data = scores / (scores + 10)
    return normalized_data


class BM25Model:
    def __init__(self, qa_path_list: list[str]):
        self.qa_path_list = qa_path_list
        self.qid_dict = {}
        self.sen2qid = OrderedDict()    # 其实无序字典亦可
        self.sentences = []
        self.retriever = None
        self.load_retriever()
        
    def load_data(self):
        # question_id,question_content,answer_content,base_name,car_type,source
        for qa_path in self.qa_path_list:
            if not os.path.exists(qa_path):
                raise Exception(f"{qa_path} not exists")
            df = pd.read_csv(qa_path, encoding="utf-8", index_col=False, keep_default_na=False)
            df['question_id'] = df['question_id'].astype(str)
            for i in range(len(df)):
                if df.loc[i,'question_id'] not in self.qid_dict:
                    self.qid_dict.setdefault(df.loc[i,'question_id'], {
                        'standard_sentence': df.loc[i,'question_content'],
                        'similar_sentence': [],
                        'answer': df.loc[i,'answer_content'],
                        'source': df.loc[i,'source'],
                        'car_type': df.loc[i,'car_type']
                    })
                else:
                    self.qid_dict[df.loc[i,'question_id']]['similar_sentence'].append(df.loc[i,'question_content'])
                self.sen2qid[df.loc[i,'question_content']] = df.loc[i,'question_id']
        self.sentences = list(self.sen2qid.keys())

    def load_retriever(self):
        self.load_data()
        corpus = self.sentences
        # Tokenize the corpus and index it
        corpus_tokens = bm25s.tokenize(corpus)
        self.retriever = bm25s.BM25(corpus=corpus, method="bm25+")
        self.retriever.index(corpus_tokens)

    def bm25_similarity(self, queries:Union[str, List[str]], topK=10):
        if not self.sentences:
            raise ValueError("corpus is None. Please add_corpus first, eg. `add_corpus(corpus)`")
        if not self.retriever:
            self.load_retriever()
        if isinstance(queries, str):
            queries = [queries]
        
        query_list = {id: query for id, query in enumerate(queries)}
        result = {query_id: {} for query_id, query in query_list.items()}

        query_tokens = bm25s.tokenize(queries)
        docs, scores = self.retriever.retrieve(query_tokens, k=topK, show_progress=False)
        for qid in range(len(queries)):
            # print(f"Best result (score: {scores[qid, 0]:.2f}): {docs[qid, 0]}")
            # print(docs[qid], scores[qid])
            # scores[qid] = min_max_normalization(scores[qid])  # 得分归一化
            # top_idx = np.argsort(scores[qid])[::-1][:topK]
            
            result[qid] = []
            for j in range(topK):
                sen = docs[qid][j]
                score = round(float(scores[qid][j]), 4)
                sen_qid = self.sen2qid.get(sen, '')
                qid_data = self.qid_dict.get(sen_qid, {})
                result[qid].append({
                    'standard_sentence': qid_data.get('standard_sentence', ''),
                    'match_sentence': sen,
                    'score': score,
                    'answer': qid_data.get('answer', ''),       # 标准问对应的答案内容
                    'source': qid_data.get('source', None),     # 数据来源（1: "FAQ知识库", 2: "一触即达意图", 4: "自定义寒暄库", 5: "内置寒暄库"）
                    'question_id': sen_qid,    # 知识id
                    'car_type': qid_data.get('car_type', None), # 适合车型
                })
        
        return list(result.values())


class Recall(object):
    def __init__(self, qa_path_list, model_type = "stella-large", is_whitening = True):
        # model_path = pretrained_model_config[model_type]
        model_path, npy_path = fine_tuned_model_config[model_type]

        self.faiss = FaissSearcher(
            model_path=model_path,
            save_npy_path=npy_path,
            qa_path_list=qa_path_list,
            index_param='Flat',  # 'HNSW64' 基于图检索，检索速度极快，且召回率几乎可以媲美Flat
            measurement='cos',   # IP
            norm_vec=True,
            is_whitening=is_whitening,  # 是否白化
        )
        self.BM25 = BM25Model(qa_path_list=qa_path_list,)
    
    def retrieve(self, text, topK, search_strategy):
        if isinstance(text, str):
            text = [text]
        # 知识检索
        if search_strategy=='hybrid':
            # 2路召回
            bm_25_hits = self.BM25.bm25_similarity(text, topK=topK*5)
            semantic_hits = self.faiss.search(text, topK=topK*5)
            merge_hits = [remove_duplicates(a+b) for a, b in zip(semantic_hits, bm_25_hits)]
            semantic_hits = [remove_duplicates(a) for a in semantic_hits]
            # results = hs_engine.search_interface_new(text, semantic_hits, merge_hits, topK=topK*10)[0]
            return merge_hits, semantic_hits
        elif search_strategy=='bm25':
            bm_25_hits = self.BM25.bm25_similarity(text, topK=topK*10)  # [0]
            bm_25_hits = [remove_duplicates(a) for a in bm_25_hits]    # 确保standard的唯一性
            return bm_25_hits
        elif search_strategy=='embedding':
            semantic_hits = self.faiss.search(text, topK=topK*10)  # [0]
            semantic_hits = [remove_duplicates(a) for a in semantic_hits]    # 确保standard的唯一性
            return semantic_hits


def remove_duplicates(docs):
    """
    保留第一个出现的，确保基于standard_sentence的唯一性
    """
    # docs = list({doc["standard_sentence"]: doc for doc in docs}.values())  # 去重?
    filter_docs = {}
    for doc in docs:
        filter_docs.setdefault(doc["standard_sentence"], doc)
    docs = list(filter_docs.values())
    return docs


if __name__ == "__main__":

    qa_path_list=[qa4api_path, qa_qy_onetouch, qa_custom_greeting, qa_global_greeting]
    recall_module = Recall(qa_path_list)

    while True:
        # 你好那个便携式充电器别的车子通用吗？
        query = input("Enter query: ")
        if query.lower() in ["exit", "quit"]:
            print("退出程序。")
            break

        try:
            merge_hits, semantic_hits = recall_module.retrieve(query, topK=5, search_strategy='hybrid')
            print(merge_hits[0][:5], semantic_hits[0][:5])

            # results = recall_module.retrieve(query, topK=5, search_strategy='bm25')
            # print(results[0][:5])

            # results = recall_module.retrieve(query, topK=5, search_strategy='embedding')
            # print(results[0][:5])

        except Exception as e:
            print(f"检索过程中发生错误：{e}")
            traceback.print_exc()
            continue  # 出现异常时，继续下一轮循环
