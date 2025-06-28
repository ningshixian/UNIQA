model_path = [
    'baichuan-inc/Baichuan2-7B-Chat',
    'ZhipuAI/chatglm2-6b',
    'qwen/Qwen-7B-Chat',
    # 量化模型的效果损失很小，但能显著降低显存占用并提升推理速度。
    'qwen/Qwen-7B-Chat-Int4',
    'qwen/Qwen-1_8B-Chat-Int4',
    '01ai/Yi-6B',

    'uer/sbert-base-chinese-nli',
    'shibing624/text2vec-base-chinese-paraphrase',
    'WangZeJun/simbert-base-chinese',
    'WangZeJun/roformer-sim-base-chinese',  # simbert V2
    'hfl/chinese-roberta-wwm-ext',
    'cyclone/simcse-chinese-roberta-wwm-ext',
    'IDEA-CCNL/Erlangshen-SimCSE-110M-Chinese',

    'moka-ai/m3e-large',
    'BAAI/bge-large-zh-v1.5',
    'infgrad/stella-large-zh-v3-1792d',
    'infgrad/stella-mrl-large-zh-v3.5-1792d', 
    'infgrad/stella-dialogue-large-zh-v3-1792d', 
    'infgrad/puff-large-v1', 
    'iampanda/zpoint_large_embedding_zh',   # 6.5 空降 C-MTEB Top1
    'Classical/Yinka',  # 基于stella-v3.5-mrl上续训
    'lier007/xiaobu-embedding-v2', 

    'maidalun/bce-reranker-base_v1', 
    'BAAI/bge-reranker-large', 
    'BAAI/bge-reranker-v2-m3', 
    "qihoo360/360Zhinao-1.8B-Reranking", 
    "BAAI/bge-reranker-v2-minicpm-layerwise", 
]

# from modelscope.hub.snapshot_download import snapshot_download
# model_idx = -5
# snapshot_download(model_path[model_idx], 
#                   cache_dir='/chj/nsx/models/',
#                   revision='master')
# exit()

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
#由于snapshot_download函数中默认的下载路径为"https://huggingface.co"，
#需将镜像网站地址设置为"https://hf-mirror.com"
#export HF_ENDPOINT=https://hf-mirror.com

# # 对于需要登录的模型，还需要两行额外代码：
# import huggingface_hub
# huggingface_hub.login("HF_TOKEN") # token 从 https://huggingface.co/settings/tokens 获取

from huggingface_hub import snapshot_download
cache_dir="/chj/nsx/models/"
# cache_dir=os.getcwd()s
model_idx = -6
# 尝试下载模型文件
snapshot_download(repo_id=model_path[model_idx],
                    repo_type='model',
                    local_dir=os.path.join(cache_dir, model_path[model_idx].split('/')[-1]),
                    cache_dir=cache_dir, 
                    local_dir_use_symlinks=False,
                    resume_download=True)
