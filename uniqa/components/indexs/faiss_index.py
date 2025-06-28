import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import sys
import re
import time
from typing import List, Union, Dict, Any, Tuple
from collections import OrderedDict
import numpy as np
from numpy import array, ndarray
import pandas as pd
import pickle
import uniqa.components.indexs.faiss_index as faiss_index
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

# from pathlib import Path
# # 获取当前文件的绝对路径
# current_dir = Path(__file__).resolve().parent
# # 获取上一级目录的路径
# parent_parent_dir = current_dir.parent
# # 将上一级目录添加到 sys.path
# sys.path.append(str(parent_parent_dir))
from uniqa.configs.config import *
from uniqa.tools.vecs_whitening import VecsWhitening


"""
faiss索引构建检索系统的全流程，支持自动whitening
参考 https://github.com/mechsihao/FaissSearcher/blob/main/backend/faiss_searcher.py
"""


class FaissIndex:
    """
    faiss索引构建检索系统的全流程
    """
    def __init__(
            self, 
            model_path: str, 
            save_npy_path: str,
            qa_path_list: list[str], 
            index_param: str = 'Flat',  # 'HNSW64' 基于图检索，检索速度极快，且召回率几乎可以媲美Flat
            measurement: str = 'cos',
            norm_vec: bool = False,
            **kwargs
    ):
        # 初始化方法
        self.qa_path_list = qa_path_list
        self.model_path = model_path
        self.save_npy_path = save_npy_path
        self.index_param = index_param
        self.measurement = measurement
        self.norm_vec = True if measurement == 'cos' else norm_vec
        self.kwargs = kwargs

        # 白化相关参数
        self.is_whitening = kwargs.get('is_whitening', False)
        self.whitening_dim = kwargs.get('whitening_dim', 128)  # 为啥越小越好？
        self.whitening_model = None

        self.qid_dict = {}  # qid: {stand, [similary], answer, source, car}
        self.sen2qid = OrderedDict()    # 知识映射qid
        self.sentences = []
        # self.answers = []
        # self.sources = []    # 数据来源
        # self.qid_list = []    # 知识id
        # self.car_type_list = []    # 适合车型

        # 向量和索引初始化
        self.index = None
        self.vec_dim = None
        self.vecs = None

        # 模型初始化
        self.model = SentenceTransformer(self.model_path, device="cuda")
        self.metric = self._set_measure_metric(measurement)

        # 构建faiss索引
        self.train()
    
    def train(self):
        """训练索引流程"""
        print(f"1、加载qa数据集并进行向量化...")
        start_time = time.time()
        self.load_data()
        self.vecs = self.get_vecs(self.sentences)
        self.vec_dim = self.vecs.shape[1]
        print(f"向量化完成，耗时: {time.time() - start_time:.2f}秒，句子数量: {len(self.sentences)}")

        print("2、构建索引...")
        start_time = time.time()
        self.vecs = self.__tofloat32__(self.vecs)
        self.__build_faiss_index()

        if not self.index.is_trained:    # 输出为True，代表该类index不需要训练，只需要add向量进去即可
            self.index.train(self.vecs)

        self.index.add(self.vecs)
        print(f"索引构建完成，耗时: {time.time() - start_time:.2f}秒，索引数据量: {self.index.ntotal}")
        
        # from sys import getsizeof
        # print(f"内存占用：{getsizeof(self.index) / 1024 / 1024} MB")

    def load_data(self):
        """2.加载和预处理QA数据"""
        for qa_path in self.qa_path_list:
            if not os.path.exists(qa_path):
                raise FileNotFoundError(f"QA数据文件不存在: {qa_path}")
            # 读取QA CSV
            # question_id,question_content,answer_content,base_name,car_type,source
            df_qa = pd.read_csv(
                qa_path, 
                encoding="utf-8", 
                index_col=False, 
                keep_default_na=False, 
                dtype={'question_id': str}  # 直接指定类型，避免后续转换
            )
            # df_qa['question_id'] = df_qa['question_id'].astype(str)

            # 批量处理数据
            for _, row in df_qa.iterrows():
                question_content = row['question_content'].strip()
                qid = row['question_id']

                # 新问题ID处理
                if qid not in self.qid_dict:
                    self.qid_dict[qid] = {
                        'standard_sentence': question_content,
                        'similar_sentence': [],
                        'answer': row['answer_content'],
                        'source': row['source'],
                        'car_type': row['car_type']
                    }

                    # 生成相似句并添加
                    eda_sentence = self._generate_similar_sentence(question_content)
                    if eda_sentence != question_content:
                        self.qid_dict[qid]['similar_sentence'].append(eda_sentence)
                        self.sen2qid[eda_sentence] = qid
                else:
                    # 已有问题ID，添加为相似句
                    self.qid_dict[qid]['similar_sentence'].append(question_content)
                
                # 建立问句到ID的映射
                self.sen2qid[question_content] = qid

        self.sentences = list(self.sen2qid.keys())
        # assert len(self.sentences) == len(self.answers) == len(self.qid_list) == len(self.car_type_list)

    def _generate_similar_sentence(self, sentence: str) -> str:
        """3.根据规则生成相似问题"""
        # 处理"理想L6/7/8"开头或“理想Mega”开头的句子
        l_car_pattern = re.compile(r'^[理想]{0,2}\s?L[6789](/L[6789])?', re.IGNORECASE)
        mega_pattern = re.compile(r'^[理想]{0,2}\s?mega', re.IGNORECASE)
        
        for pattern in [l_car_pattern, mega_pattern]:
            match = pattern.match(sentence)
            if match:
                prefix = match.group()
                # 移除前缀、标点符号，并调整位置
                modified = re.sub(prefix, '', sentence, flags=re.IGNORECASE)
                modified = re.sub(r'[。?？！]$', '', modified)
                modified = modified + re.sub('理想', '', prefix, flags=re.IGNORECASE)
                return modified.strip()
                
        return sentence.strip()

    def get_vecs(self, items: Union[List[str], ndarray]) -> array:
        """4.将文本转换为向量表示"""
        if not items:
            raise ValueError("输入项不能为空")
        
        try:
            vecs = self.model.encode(items)  # normalize_embeddings=self.norm_vec
            # 向量白化
            if self.is_whitening:
                if not self.whitening_model:    # 只会在一开始的train中执行一次
                    self.whitening_model = self.__init_whitening_model(whitening_source_vecs=np.array(vecs))
                    if self.whitening_model is None:
                        raise ValueError("白化模型初始化失败")
                vecs = self.whitening_model.transform(vecs)
            # 向量归一化
            if self.norm_vec:
                vecs = self.__normvec__(vecs)
                # faiss.normalize_L2(vecs)
            
            # 转换为float32类型
            return self.__tofloat32__(vecs)

        except Exception as e:
            # 这里可以记录日志或执行其他错误处理策略
            raise RuntimeError(f"向量处理失败: {str(e)}") from e
    
    @staticmethod
    def __tofloat32__(vecs):
        return vecs.astype(np.float32)

    @staticmethod
    def __normvec__(vecs):
        return vecs / (vecs ** 2).sum(axis=1, keepdims=True) ** 0.5

    def __build_faiss_index(self):
        """5.构建Faiss索引"""
        if 'hnsw' in self.index_param.lower() and ',' not in self.index_param:
            # 提取HNSW参数
            hnsw_size = int(self.index_param.lower().split('hnsw')[-1])
            self.index = faiss_index.IndexHNSWFlat(self.vec_dim, hnsw_size, self.metric)
        else:
            self.index = faiss_index.index_factory(self.vec_dim, self.index_param, self.metric)
        
        self.index.verbose = True
        self.index.do_polysemous_training = False
        return self

    def search(self, target: Union[List[str], ndarray], topK: Union[int, List[int]]):
        """6.向量检索功能实现"""
        # 生成查询向量
        target_vec: ndarray = self.get_vecs(target)

        # 执行检索
        distances, indices = self.index.search(target_vec, topK)

        # # 排除indices中standard_sentence相同的索引
        # indices = [i for indice in indices for i in indice if self.sentences[indices[i][j]]]
        
        # 处理检索结果
        closest_matches = []
        for i, (dist_row, idx_row) in enumerate(zip(distances, indices)):
            matches = []
            for j, (score, idx) in enumerate(zip(dist_row, idx_row)):
                # 获取对应句子与问题ID
                sen = self.sentences[idx]
                sen_qid = self.sen2qid.get(sen, '')
                qid_data = self.qid_dict.get(sen_qid, {})
                
                # 构建返回结果
                matches.append({
                    'standard_sentence': qid_data.get('standard_sentence', ''),
                    'match_sentence': sen,
                    'score': round(float(score), 4),
                    'answer': qid_data.get('answer', ''),       # 标准问对应的答案内容
                    'source': qid_data.get('source'),           # 数据来源（1: "FAQ知识库", 2: "一触即达意图", 4: "自定义寒暄库", 5: "内置寒暄库"）
                    'question_id': sen_qid,                     # 知识id
                    'car_type': qid_data.get('car_type')        # 适合车型
                })
            closest_matches.append(matches)

        # print(closest_matches)
        return closest_matches


    @staticmethod
    def _set_measure_metric(measurement: str) -> int:
        metric_dict = {
            'cos': faiss_index.METRIC_INNER_PRODUCT,
            'l1': faiss_index.METRIC_L1,
            'l2': faiss_index.METRIC_L2,
            'l_inf': faiss_index.METRIC_Linf,
            'l_p': faiss_index.METRIC_Lp,
            'brayCurtis': faiss_index.METRIC_BrayCurtis,
            'canberra': faiss_index.METRIC_Canberra,
            'jensen_shannon': faiss_index.METRIC_JensenShannon
        }
        if measurement in metric_dict:
            return metric_dict[measurement]
        else:
            supported = ', '.join(metric_map.keys())
            raise ValueError(f"不支持的度量方式: '{measurement}'，支持的度量方式: [{supported}]")

    def __init_whitening_model(self, **kwargs):
        """简单的向量白化改善句向量质量 
        https://github.com/bojone/BERT-whitening
        """
        if self.is_whitening:
            whitening_model = VecsWhitening(n_components=self.whitening_dim)
            if 'whitening_path' in kwargs:
                whitening_model_path = kwargs['whitening_path']
                whitening_model.load_bw_model(whitening_model_path)
            elif 'whitening_source_vecs' in kwargs:
                _start_time = time.time()
                whitening_model.fit(kwargs['whitening_source_vecs'])
                # print(f"Whitening model build cost time: {time.time() - _start_time}")
            else:
                whitening_model = None

            return whitening_model
        else:
            return None

    # 新增实用功能

    def get_index_stats(self) -> Dict:
        """获取索引统计信息"""
        return {
            'index_type': self.index_param,
            'total_vectors': self.index.ntotal if self.index else 0,
            'dimension': self.vec_dim,
            'measurement': self.measurement,
            'whitening': self.is_whitening,
            'whitening_dim': self.whitening_dim if self.is_whitening else None
        }

    def batch_search_with_progress(
        self, 
        queries: List[str], 
        topK: int = 5, 
        batch_size: int = 32, 
        show_progress: bool = True
        ) -> Tuple[List[List[Dict]], Dict]:
        """
        带进度反馈的批量检索功能
        
        参数:
            queries: 待检索的查询文本列表
            topK: 每个查询返回的最相似结果数量
            batch_size: 批处理大小，用于控制内存使用
            show_progress: 是否显示进度条
            
        返回:
            (检索结果列表, 性能统计信息)
        """
        # 参数验证与前面相同...
        
        # 结果与统计信息
        all_results = []
        stats = {
            'total_queries': len(queries),
            'completed_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'total_time': 0,
            'encoding_time': 0,
            'search_time': 0,
            'processing_time': 0
        }
        
        total_start_time = time.time()
        total_queries = len(queries)
        
        # 进度条设置
        if show_progress:
            try:
                from tqdm import tqdm
                pbar = tqdm(total=total_queries, desc="批量检索进度")
            except ImportError:
                print("提示: 安装tqdm包可以显示进度条")
                pbar = None
                show_progress = False
        
        try:
            # 批处理逻辑
            for i in range(0, total_queries, batch_size):
                batch_end = min(i + batch_size, total_queries)
                current_batch = queries[i:batch_end]
                batch_size_actual = len(current_batch)
                
                # 向量编码计时
                encoding_start = time.time()
                try:
                    batch_vecs = self.get_vecs(current_batch)
                    stats['encoding_time'] += time.time() - encoding_start
                    
                    # 检索计时
                    search_start = time.time()
                    distances, indices = self.index.search(batch_vecs, topK)
                    stats['search_time'] += time.time() - search_start
                    
                    # 结果处理计时
                    processing_start = time.time()
                    
                    batch_results = []
                    for j, (dist_row, idx_row) in enumerate(zip(distances, indices)):
                        matches = []
                        valid_matches = 0
                        
                        for k, (score, idx) in enumerate(zip(dist_row, idx_row)):
                            # 跳过无效索引
                            if idx < 0 or idx >= len(self.sentences):
                                continue
                                
                            # 获取并组装结果
                            sen = self.sentences[idx]
                            sen_qid = self.sen2qid.get(sen, '')
                            qid_data = self.qid_dict.get(sen_qid, {})
                            
                            match_dict = {
                                'query': current_batch[j],
                                'standard_sentence': qid_data.get('standard_sentence', ''),
                                'match_sentence': sen,
                                'score': round(float(score), 4),
                                'answer': qid_data.get('answer', ''),
                                'source': qid_data.get('source'),
                                'question_id': sen_qid,
                                'car_type': qid_data.get('car_type'),
                                'rank': valid_matches + 1  # 添加排名信息
                            }
                            matches.append(match_dict)
                            valid_matches += 1
                        
                        # 确保结果按相似度排序
                        is_similarity = self.measurement == 'cos'  # 余弦相似度是越大越相似
                        sorted_matches = sorted(matches, key=lambda x: x['score'], reverse=is_similarity)
                        batch_results.append(sorted_matches)
                        
                        # 更新统计
                        stats['successful_queries'] += 1
                    
                    # 合并批次结果
                    all_results.extend(batch_results)
                    stats['processing_time'] += time.time() - processing_start
                    
                except Exception as e:
                    print(f"批次 {i//batch_size + 1} 处理错误: {str(e)}")
                    # 为失败的查询添加空结果
                    for _ in range(batch_size_actual):
                        all_results.append([])
                        stats['failed_queries'] += 1
                
                # 更新进度
                stats['completed_queries'] += batch_size_actual
                if show_progress and pbar:
                    pbar.update(batch_size_actual)
            
            # 清理进度条
            if show_progress and pbar:
                pbar.close()
            
            # 总时间计算
            stats['total_time'] = time.time() - total_start_time
            
            return all_results, stats
            
        except Exception as e:
            if show_progress and pbar:
                pbar.close()
            print(f"批量检索过程中发生严重错误: {str(e)}")
            stats['total_time'] = time.time() - total_start_time
            return all_results, stats

    def update_index(self, new_sentences: List[str], new_question_ids: List[str], 
            new_answers: List[str], new_sources: List[str] = None, 
            new_car_types: List[str] = None) -> None:
        """更新索引(添加新数据)"""
        pass
    
    def refresh_index_from_file(self, updated_path: str, preserve_custom_similars: bool = True) -> Dict:
        """从更新的QA文件完全重建索引
        参数:
            updated_path: 更新的QA文件路径
        """
        try:
            new_df = pd.read_csv(
                updated_path, 
                encoding="utf-8", 
                index_col=False, 
                keep_default_na=False,
                dtype={'question_id': str}
            )
        except Exception as e:
            raise IOError(f"读取QA文件失败: {str(e)}")
        
        new_qid_dict = {}
        new_sen2qid = OrderedDict()
        new_sentences = []
        
        # 步骤3: 提取更新后文件中的问题ID集合
        new_file_qids = set(new_df['question_id'].unique())

        # 处理需要更新的问题ID
        for qid in new_file_qids:
            rows = new_df[new_df['question_id'] == qid]
            if len(rows) > 0:
                row = rows.iloc[0]  # qid第一行为标准问
                # 获取相似问题
                similar_questions = [row['question_content'] for _, row in rows.iloc[1:].iterrows()]
                # 更新问题
                self.qid_dict[qid] = {
                    'standard_sentence': row['question_content'].strip(),
                    'similar_sentence': similar_questions,
                    'answer': row['answer_content'],
                    'source': row.get("source", "知识库"),
                    'car_type': row['car_type']
                }

                # 之后还得做 diff，找到被修改的内容后再能准确索引，太麻烦！
                self.sen2qid.update(...)
                self.sentences.extend(...)
                new_vecs = self.get_vecs(...)
                new_vecs = faq_sys.recall_module.faiss.__tofloat32__(new_vecs)
                self.index.add(new_vecs)
                pass

    # 其他

    def save_vecs(self):
        np.save(self.save_npy_path, self.vecs)
        print("保存embedding →→→ 文件data.x.npy")

    def save_searcher(self, path):
        file = open(path, "wb")
        pickle.dump(self, file)
        file.close()

    @staticmethod
    def load_searcher(path):
        file = open(path, "rb")
        return pickle.load(file)

    def save_index(self, index_save_path):
        faiss_index.write_index(self.index, index_save_path)

    def load_index(self, index_path):
        print(f"Load index...")
        self.index = faiss_index.read_index(index_path)
        # assert self.index.ntotal == len(self.items), f"Index sample nums {self.index.ntotal} != Items length {len(self.items)}"
        # assert self.index.d == self.vec_dim, f"Index dim {self.index.d} != Vecs dim {self.vec_dim}"
        # assert self.index.is_trained, "Index dose not trained"

    def cal_sim(self, item1: str, item2: str):
        """
        计算两个句子嵌入向量的余弦相似度。
        """
        # 验证输入是否为非空字符串
        if not isinstance(item1, str) or not item1.strip():
            raise ValueError("item1必须是非空字符串")
        if not isinstance(item2, str) or not item2.strip():
            raise ValueError("item2必须是非空字符串")

        vec1 = self.get_vecs([item1])
        vec2 = self.get_vecs([item2])

        # 计算并返回点积和余弦相似度
        return np.dot(vec1, vec2.T), cosine_similarity(vec1, vec2)

    def rank_items_by_similarity(self, item1: str, items2: list) -> pd.DataFrame:
        """根据相似程度对items2列表中的项进行排序"""
        vec1 = self.get_vecs([item1])
        vecs = self.get_vecs(items2)
        sim_score_list = vec1.dot(vecs.T)
        sim_df = pd.DataFrame(items2, columns=['item'])
        sim_df['score'] = sim_score_list
        return sim_df.sort_values(by="score", ascending=False)


if __name__ == "__main__":

    topK = 5
    is_whitening = True

    model_type = "stella-large"
    # pretrained_model_path = pretrained_model_config[model_type]
    save_ft_model_path, npy_path = fine_tuned_model_config[model_type]

    searcher = FaissSearcher(
        model_path=save_ft_model_path,
        save_npy_path=npy_path,
        qa_path_list=[qa4api_path, qa_qy_onetouch, qa_custom_greeting, qa_global_greeting],
        index_param='Flat',  # 'HNSW64' 基于图检索，检索速度极快，且召回率几乎可以媲美Flat
        measurement='cos',   # IP
        norm_vec=True,
        is_whitening=is_whitening
    )
    # searcher.train()
    # searcher.save_searcher("data_output/searcher.abc")    #1.3G
    # print(searcher.search(["售后服务-服务中心信息查询"], topK=topK))  # 更新知识后，记得执行 knowledge_process.py
    print(searcher.get_index_stats())

    badcases = [
        "方向盘加热", "后挡风玻璃总是有雾", "座椅怎么升高", "驾驶员内监控关闭", 
        "怎么切换驾驶模式", "理想车内有没监控", "座椅加热怎么开？（MEGA）"
    ]
    while True:
        s1 = input("s1: ")
        s2 = input("s2: ")
        if s1.lower() in ['exit', 'quit']:
            print("退出程序。")
            break

        start_time = time.time()
        try:
            res = searcher.search([s1, s2], topK)
        except Exception as e:
            print(f"检索过程中发生错误：{e}")
            continue  # 出现异常时，继续下一轮循环
        
        func = lambda x: "\n".join([f"{r['score']:.4f}\t{r['standard_sentence']}" for r in x])
        res = list(map(func, res))
        print("Retrieve:\n", res)

        end_time = time.time()
        print("函数运行时间为：", end_time - start_time, "秒")

        try:
            print("Similarity:", searcher.cal_sim(s1, s2))
        except Exception as e:
            print(f"计算相似度时发生错误：{e}")

