import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
import numpy as np
import pandas as pd

from pathlib import Path
# 获取当前文件的绝对路径
current_dir = Path(__file__).resolve().parent
# 获取上一级目录的路径
parent_parent_dir = current_dir.parent
# 将上一级目录添加到 sys.path
sys.path.append(str(parent_parent_dir))
from configs.config import *
from uniqa.components.preprocessors import cleaning_func

from uniqa.components.indexs.faiss_index import FaissSearcher
# from bm25 import BM25Model
# from hybrid_retriever import HybridSearchEngine
from training.bm25.bm25_sparse import BM25Model
from uniqa.components.rank import PostRank

"""
新-模型评估脚本
- 融合了最初的800+数据，以及7.10新标注的 1700+数据
- 标注结果更新：可对应多个标注答案（只要命中一个正确答案，则认为预测正确）
- 模型评估依旧是基于top-k的准确率，以「knowledge_id」进行评估

cd evals & python evaluate_hybrid_new.py
"""


class Evaluator:

    def __init__(self, top_k, save_ft_model_path, npy_path, is_whitening=False):
        self.top_k = top_k
        # self.thredshold = thredshold
        self.save_ft_model_path = save_ft_model_path
        self.npy_path = npy_path
        self.searcher = FaissSearcher(
            model_path=save_ft_model_path,
            save_npy_path=npy_path,
            qa_path_list=[qa_pair_path],
            index_param='Flat',  # 'HNSW64' 基于图检索，检索速度极快，且召回率几乎可以媲美Flat
            measurement='cos',   # IP
            norm_vec=True,
            is_whitening=is_whitening,  # 是否白化
        )
        # self.searcher.train()
        self.BM25 = BM25Model(qa_path_list=[qa_pair_path],)
        # self.hs_engine = HybridSearchEngine("../models/bge-reranker-large")    # 最终 sigmoid 输出的分值普遍较高
        self.hs_engine = PostRank(rerank_model_path)

    def remove_duplicates(self, docs):
        """
        保留第一个出现的，确保基于standard_sentence的唯一性
        """
        # docs = list({doc["standard_sentence"]: doc for doc in docs}.values())  # 去重?
        filter_docs = {}
        for doc in docs:
            filter_docs.setdefault(doc["standard_sentence"], doc)
        docs = list(filter_docs.values())
        return docs
    
    def retrieve(self, questions):
        # 纯embedding召回
        semantic_hits = self.searcher.search(questions, topK=self.top_k*10)
        retrieve_results = [self.remove_duplicates(a) for a in semantic_hits]    # 确保standard的唯一性
        retrieve_results = [b[:self.top_k] for b in retrieve_results]

        # # 2路召回 + rerank
        # bm_25_hits = self.BM25.bm25_similarity(questions, self.top_k*5)
        # semantic_hits = self.searcher.search(questions, self.top_k*5)
        # doc_hits = [self.remove_duplicates(a+b) for a, b in zip(semantic_hits, bm_25_hits)]
        # # retrieve_results = self.hs_engine.search_interface(questions, doc_hits, topK=self.top_k)    #[0]
        # retrieve_results = self.hs_engine.rerank(questions, merge_hits, semantic_hits, fusion=False)
        # retrieve_results = retrieve_results[0][:self.top_k]

        # # 2路召回 + rerank + 线性加权融合
        # bm_25_hits = self.BM25.bm25_similarity(questions, self.top_k*5)
        # semantic_hits = self.searcher.search(questions, self.top_k*5)
        # merge_hits = [self.remove_duplicates(a+b) for a, b in zip(semantic_hits, bm_25_hits)]
        # semantic_hits = [self.remove_duplicates(a) for a in semantic_hits]
        # retrieve_results = self.hs_engine.rerank(questions, merge_hits, semantic_hits, fusion=True)
        # retrieve_results = retrieve_results[0][:self.top_k]

        return retrieve_results

    def evaluate_top_k_accuracy(self, retrieve_results, questions, standard_answers, w_path, thredshold):
        # 评估模型的预测结果（top-k）
        t1_count, t2_count, t3_count, t5_count, t10_count = 0, 0, 0, 0, 0
        t1_flags, t2_flags, t3_flags, t5_flags, t10_flags = [["✖"] * len(questions) for _ in range(5)]
        predicts = []
        
        for i, (retrieve_result, standard_answer) in enumerate(zip(retrieve_results, standard_answers)):
            predicts.append(
                "\n".join([format(item['score'], '.4f') +"\t"+ item['standard_sentence'] for item in retrieve_result])
            )
            # candidates = [x['standard_sentence'] for x in retrieve_result if x["score"] > thredshold]
            # if not candidates:  # 无答案情况
            #     candidates = ["暂无标准问"]     # ['-']
            candidates = [x['question_id'].strip() for x in retrieve_result if x["score"] > thredshold]
            if not candidates:  # 无答案情况
                candidates = [""] # 根据 kid 匹配

            if set(candidates[:1]) & set(standard_answer):
                t1_count += 1
                t1_flags[i] = "✔"
            if set(candidates[:2]) & set(standard_answer):
                t2_count += 1
                t2_flags[i] = "✔"
            if set(candidates[:3]) & set(standard_answer):
                t3_count += 1
                t3_flags[i] = "✔"
            if set(candidates[:5]) & set(standard_answer):
                t5_count += 1
                t5_flags[i] = "✔"
            if set(candidates[:10]) & set(standard_answer):
                t10_count += 1
                t10_flags[i] = "✔"

        # 计算准确度
        top_1_accuracy = t1_count / len(questions)
        top_2_accuracy = t2_count / len(questions)
        top_3_accuracy = t3_count / len(questions)
        top_5_accuracy = t5_count / len(questions)
        top_10_accuracy = t10_count / len(questions)
        print(f"评测结果如下：({thredshold})")
        print(f"Top-1 Accuracy: {top_1_accuracy:.2%}")
        print(f"Top-2 Accuracy: {top_2_accuracy:.2%}")
        print(f"Top-3 Accuracy: {top_3_accuracy:.2%}")
        print(f"Top-5 Accuracy: {top_5_accuracy:.2%}")
        print(f"Top-10 Accuracy: {top_10_accuracy:.2%}")

        # # 计算准召
        # precision_1 = precision_score(["✔"] * len(questions), t1_flags, average='macro')
        # recall_1 = recall_score(["✔"] * len(questions), t1_flags, average='macro')
        # print(precision_1, recall_1)    # 0.16395 0.5

        _df = pd.DataFrame({
            'hit@1/2/3/5/10': zip(t1_flags, t2_flags,t3_flags,t5_flags,t10_flags),
            'question': questions,
            'answer': standard_answers,
            'predicts': predicts,
        })
        _df.to_csv(w_path, index=False)

        return f"{top_1_accuracy:.2%},{top_2_accuracy:.2%},{top_3_accuracy:.2%},{top_5_accuracy:.2%},{top_10_accuracy:.2%}"


def do_execute(test_path, evaluator):
    """执行评测的入口"""
    questions, questions_drop = [], []
    standard_answers, standard_answers_drop = [], []
    df = pd.read_csv(test_path, header=0, index_col=False, encoding='utf-8', keep_default_na=False)
    for i, row in df.iterrows():
        msg = cleaning_func.clean_text(row["query"])
        answer = row["knowledge_id"].split("\n")
        answer = [x.strip() for x in answer if x.strip()]
        if not answer:
            answer = [""]
        questions.append(msg)
        standard_answers.append(answer)
        if row["knowledge_id"].strip():
            questions_drop.append(msg)
            standard_answers_drop.append(answer)

    print("评测集A数据：", len(questions))  # 2616
    print("评测集B数据：", len(questions_drop)) # 2023
    retrieve_results = evaluator.retrieve(questions)
    retrieve_results_drop = evaluator.retrieve(questions_drop)

    writer = open(testset_folder+"/hit.csv", 'w')
    writer.write("阈值,hit@1,hit@2,hit@3,hit@5,hit@10")

    # 计算top准确率
    for thredshold in [0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.5, 0.4, 0.3]:
        from configs.config import eval_path, eval_path_filter
        eval_path = eval_path.format(model_type, thredshold)
        eval_path_filter = eval_path_filter.format(model_type, thredshold)
        scores = evaluator.evaluate_top_k_accuracy(retrieve_results, questions, standard_answers, eval_path, thredshold)
        scores_drop = evaluator.evaluate_top_k_accuracy(retrieve_results_drop, questions_drop, standard_answers_drop, eval_path_filter, thredshold)
        writer.write(f"\n{thredshold}," + scores)
        writer.write(f"\n{thredshold}," + scores_drop)


if __name__ == "__main__":

    top_k = 10
    is_whitening = True
    model_type = ["stella-large", "xiaobu", "Conan-embedding"][0]
    # config.test_set_path = "data_output/testset_new/testset_update.csv"     # 评测集

    # model_path, npy_path = pretrained_model_config[model_type], None
    # model_path, npy_path = fine_tuned_model_config[model_type]
    model_path, npy_path = "../finetuned_models/stella-fine-tuned-v4.5", None
    
    # rerank_model_path = "/chj/nsx/models/bge-reranker-v2-m3"
    rerank_model_path = "/chj/nsx/models/bge-reranker-large"

    evaluator = Evaluator( top_k,  model_path,  npy_path, is_whitening )
    do_execute(test_set_path, evaluator)
