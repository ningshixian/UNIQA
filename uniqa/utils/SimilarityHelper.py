import numpy as np
from scipy.optimize import linprog
import textdistance
from fuzzywuzzy import fuzz
import difflib

"""
文本（句子）相似度计算
句子模糊匹配
"""


def hamming_sim(text1, text2):
    """
    计算两个字符串之间的汉明距离
    """
    # textdistance.hamming('text','test')
    # 将编辑距离限制在0和1之间
    score = textdistance.hamming.normalized_similarity(text1, text2)
    return score


def levenshtein_sim(text1, text2):
    """
    计算两个字符串之间的Levenshtein距离
    """
    # textdistance.levenshtein("this test", "that test") # 2
    score = textdistance.levenshtein.normalized_similarity(text1, text2)  # 0.8
    return score


def jw_sim(text1, text2):
    """
    计算两个字符串之间的Jaro-Winkler距离
    """
    score = textdistance.jaro_winkler(text1, text2)  # 0.8
    return score


def cosine_string_sim(text1, text2):
    """
    计算两个字符串之间的余弦相似度
    """
    # score = textdistance.cosine(text1, text2)  # 0.8
    score = textdistance.cosine.normalized_similarity(text1, text2)
    return score


def cosine_sim(vector_a, vector_b):
    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a
    :param vector_b: 向量 b
    :return: sim
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)  # np.dot(vector1,vector2)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos  # 修正余弦相似度
    return sim


def corrected_cosine_sim(vector_a, vector_b):
    """
    计算两个向量之间的修正余弦相似度
    """
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    cosine_similarity = dot_product / (norm_a * norm_b)
    shorter_length = min(len(vector_a), len(vector_b))
    longer_length = max(len(vector_a), len(vector_b))
    corrected_similarity = cosine_similarity * (shorter_length / longer_length)
    return corrected_similarity


def cosine_similarity(matrix1, matrix2):
    """
    计算两个矩阵的余弦相似度
    """
    dot_product = np.dot(matrix1, matrix2.T)
    norm1 = np.linalg.norm(matrix1, axis=1, keepdims=True)
    norm2 = np.linalg.norm(matrix2, axis=1, keepdims=True)
    similarity = dot_product / (norm1 * norm2.T)
    return similarity


def cosine_similarity_sklearn(matrix1, matrix2):
    """
    计算两个矩阵的修正余弦相似度
    """
    from sklearn.metrics.pairwise import cosine_similarity
    similarity = cosine_similarity(np.array(matrix1),np.array(matrix2))      
    return similarity


def EuclideanDistance(vec1, vec2):
    """
    计算两个向量之间的欧式距离
    """
    score = 1.0 / (1.0 + np.linalg.norm(vec1 - vec2, ord=2))
    # score = 1.0/(1.0 + np.sqrt(np.sum(np.square(vec1-vec2))))
    return score


def Manhattan(vec1, vec2):
    return np.linalg.norm(vec1 - vec2, ord=1)
    # return 1/(1+np.sqrt((np.sum(vec1-vec2)**2)))


def Pearson_sim(vec1, vec2):
    """
    计算两个向量之间的皮尔森相关系数
    """
    X = np.vstack([vec1, vec2])
    return 0.5 + 0.5 * np.corrcoef(X)[0][1]


def wasserstein_distance(p, q, D):
    """通过线性规划求Wasserstein距离
    p.shape=[m], q.shape=[n], D.shape=[m, n]
    p.sum()=1, q.sum()=1, p∈[0,1], q∈[0,1]
    """
    A_eq = []
    for i in range(len(p)):
        A = np.zeros_like(D)
        A[i, :] = 1
        A_eq.append(A.reshape(-1))
    for i in range(len(q)):
        A = np.zeros_like(D)
        A[:, i] = 1
        A_eq.append(A.reshape(-1))
    A_eq = np.array(A_eq)
    b_eq = np.concatenate([p, q])
    D = D.reshape(-1)
    result = linprog(D, A_eq=A_eq[:-1], b_eq=b_eq[:-1])
    return result.fun


def word_mover_distance(x, y):
    """WMD（Word Mover's Distance）的参考实现
    x.shape=[m,d], y.shape=[n,d]
    """
    p = np.ones(x.shape[0]) / x.shape[0]
    q = np.ones(y.shape[0]) / y.shape[0]
    D = np.sqrt(np.square(x[:, None] - y[None, :]).mean(axis=2))
    return wasserstein_distance(p, q, D)


def word_rotator_distance(x, y):
    """WRD（Word Rotator's Distance）的参考实现
    x.shape=[m,d], y.shape=[n,d]
    """
    x_norm = (x ** 2).sum(axis=1, keepdims=True) ** 0.5
    y_norm = (y ** 2).sum(axis=1, keepdims=True) ** 0.5
    p = x_norm[:, 0] / x_norm.sum()
    q = y_norm[:, 0] / y_norm.sum()
    D = 1 - np.dot(x / x_norm, (y / y_norm).T)
    return wasserstein_distance(p, q, D)


def word_rotator_similarity(x, y):
    """1 - WRD
    x.shape=[m,d], y.shape=[n,d]
    """
    return 1 - word_rotator_distance(x, y)


def fuzz_sim(s1_ist, s2_list):
    """
    模糊匹配相似度
    a	str	句子1
    b	str	句子2
    """
    results = []
    for s1 in s1_ist:
        for s2 in s2_list:
            similar_score = fuzz.ratio(s1, s2)
            res = {"score": similar_score, "sentences": {"s1": s1, "s2": s2}}
            results.append(res)
    return results


def matches(large_string, query_string, threshold=0.86):
    """
    使用方法：
    1. 利用difflib计算差异工具的 `SequenceMatcher` 方法，可以找到不包含‘垃圾’元素的最长连续匹配子序列 `match`;
    2. 然后将相同的思想递归应用于匹配子序列左侧和右侧的序列片段;
    3. 计算 `match` 和 `query` 的余弦距离，超过阈值即可将 `match` 作为query候选；
    """
    s = difflib.SequenceMatcher(None, large_string, query_string)
    match = ''.join(large_string[i:i+n] for i, j, n in s.get_matching_blocks() if n)
    if match:
        ratio1 = cosine_string_sim(match, query_string)
        ratio1 = round(ratio1, 5)
        if ratio1 >= threshold:
            print("模糊得分：", ratio1, "\t最长匹配子串：", match, "\t查询词：", query_string)
            return match, ratio1
    return '', 0


# print(matches("在线客服 商场", "C2在线客服", threshold=0.80))


# def fast_fuzzy_matching():
#     import pandas as pd
#     from matplotlib import style
#     style.use('fivethirtyeight')
#     import re
#     import time
#     from ftfy import fix_text
#     import numpy as np
#     from scipy.sparse import csr_matrix
#     import sparse_dot_topn.sparse_dot_topn as ct
#     from sklearn.feature_extraction.text import TfidfVectorizer

#     def ngrams(string, n=3):
#         string = fix_text(string) # fix text
#         string = string.encode("ascii", errors="ignore").decode() #remove non ascii chars
#         string = string.lower()
#         chars_to_remove = [")","(",".","|","[","]","{","}","'"]
#         rx = '[' + re.escape(''.join(chars_to_remove)) + ']'
#         string = re.sub(rx, '', string)
#         string = string.replace('&', 'and')
#         string = string.replace(',', ' ')
#         string = string.replace('-', ' ')
#         string = string.title() # normalise case - capital at start of each word
#         string = re.sub(' +',' ',string).strip() # get rid of multiple spaces and replace with a single
#         string = ' '+ string +' ' # pad names for ngrams...
#         string = re.sub(r'[,-./]|\sBD',r'', string)
#         ngrams = zip(*[string[i:] for i in range(n)])
#         return [''.join(ngram) for ngram in ngrams]

#     def awesome_cossim_top(A, B, ntop, lower_bound=0):
#         # force A and B as a CSR matrix.
#         # If they have already been CSR, there is no overhead
#         A = A.tocsr()
#         B = B.tocsr()
#         M, _ = A.shape
#         _, N = B.shape
#         idx_dtype = np.int32
#         nnz_max = M*ntop
#         indptr = np.zeros(M+1, dtype=idx_dtype)
#         indices = np.zeros(nnz_max, dtype=idx_dtype)
#         data = np.zeros(nnz_max, dtype=A.dtype)

#         ct.sparse_dot_topn(
#             M, N, np.asarray(A.indptr, dtype=idx_dtype),
#             np.asarray(A.indices, dtype=idx_dtype),
#             A.data,
#             np.asarray(B.indptr, dtype=idx_dtype),
#             np.asarray(B.indices, dtype=idx_dtype),
#             B.data,
#             ntop,
#             lower_bound,
#             indptr, indices, data)

#         # return (data,indices,indptr)
#         return csr_matrix((data,indices,indptr),shape=(M,N))    # .toarray().T    # .A

#     def get_matches_df(sparse_matrix, name_vector, top=100):
#         non_zeros = sparse_matrix.nonzero()
#         sparserows = non_zeros[0]
#         sparsecols = non_zeros[1]
        
#         if top:
#             nr_matches = top
#         else:
#             nr_matches = sparsecols.size
        
#         left_side = np.empty([nr_matches], dtype=object)
#         right_side = np.empty([nr_matches], dtype=object)
#         similairity = np.zeros(nr_matches)
        
#         for index in range(0, nr_matches):
#             left_side[index] = name_vector[sparserows[index]]
#             right_side[index] = name_vector[sparsecols[index]]
#             similairity[index] = sparse_matrix.data[index]
        
#         return pd.DataFrame({'left_side': left_side,
#                             'right_side': right_side,
#                             'similairity': similairity})


#     print('All 3-grams in "Department":')
#     print(ngrams('Department'))
    
#     names =  pd.read_csv('messy org names.csv',encoding='latin')
#     print('The shape: %d x %d' % names.shape)
#     print('There are %d unique values' % names.buyer.shape[0])
    
#     org_names = names['buyer'].unique()
#     vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams)
#     tf_idf_matrix = vectorizer.fit_transform(org_names)
#     # print(tf_idf_matrix[0])

#     t1 = time.time()
#     matches = awesome_cossim_top(tf_idf_matrix, tf_idf_matrix.transpose(), 10, 0.85)
#     t = time.time()-t1
#     print("SELFTIMED:", t)

#     matches_df = get_matches_df(matches, org_names, top=10)
#     matches_df = matches_df[matches_df['similairity'] < 0.99999] # Remove all exact matches
#     print(matches_df.sample())
#     print(matches_df.sort_values(['similairity'], ascending=False).head(10))

