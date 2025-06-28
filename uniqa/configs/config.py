import os 
env = os.getenv("FAQ_ENV", "")

if env=='prod': #生产环境
    path_root = os.getcwd()
    # fine_tuned_folder = "/lpai/inputs/models/faq-for-dialogue-24-06-17-1"
    fine_tuned_folder = "/lpai/inputs/models/faq-for-dialogue-24-09-18-1"
    kafka_data_folder = os.path.join(path_root, "data_factory/kafka_prod")
    model_cache_folder = "/lpai/inputs/models/faq-for-dialogue-24-06-17-2"
else: #测试环境
    # path_root = os.getcwd()
    path_root = "/chj/nsx/faq-semantic-retrieval"
    fine_tuned_folder = os.path.join(path_root, "finetuned_models")
    kafka_data_folder = os.path.join(path_root, "data_factory/kafka_test")
    model_cache_folder = "/chj/nsx/models"


# 预训练模型 / 预训练embedding
pretrained_model_config = {
    'distilbert': model_cache_folder+'/distilbert-multilingual-nli-stsb-quora-ranking',    # ✔
    'paraphrase-multilingual-MiniLM-L12-v2': 'paraphrase-multilingual-MiniLM-L12-v2',

    'sbert-base': model_cache_folder+'/sbert-base-chinese-nli',    # ✔
    'sbert-chinese-general-v2': 'DMetaSoul/sbert-chinese-general-v2',    # ✔
    'm3e-large': model_cache_folder + '/m3e-large',
    'bge-large': '/chj/nsx/embeddings/bge-large-zh-v1.5',
    "gte-large": model_cache_folder + '/gte-large-zh',
    "stella-large": model_cache_folder+"/stella-large-zh-v3-1792d",
    "puff-large": model_cache_folder+"/puff-large-v1",
    "xiaobu": model_cache_folder+"/xiaobu-embedding-v2",
    
    'roberta': model_cache_folder+ '/chinese-roberta-wwm-ext',
    'simcse-roberta': model_cache_folder+ '/simcse-chinese-roberta-wwm-ext',
    'erlangshen': model_cache_folder+ '/Erlangshen-SimCSE-110M-Chinese',
    'text2vec': model_cache_folder + '/text2vec-base-chinese-paraphrase',    # ✔
    'simbert': model_cache_folder+"/simbert-base-chinese", 
}

npy_path = os.path.join(path_root,  "/evals/data.npy")

# 微调之后的模型
fine_tuned_model_config = {
    "xiaobu": [
        fine_tuned_folder + "/xiaobu-fine-tuned", 
        os.path.join(path_root,  "evals/data.xiaobu.npy")
    ], 
    "puff-large": [
        fine_tuned_folder + "/puff-fine-tuned", 
        os.path.join(path_root,  "evals/data.puff.npy")
    ], 
    "stella-large": [
        # fine_tuned_folder + "/stella-fine-tuned-v3",    # v3 v4 v4.1
        fine_tuned_folder + "/stella-fine-tuned-v4.5", 
        os.path.join(path_root,  "evals/data.stella.npy")
    ], 
    "sbert-base": [
        fine_tuned_folder + "/sbert-fine-tuned",
        os.path.join(path_root,  "evals/data.sbert.npy")
    ], 
    "text2vec": [
        fine_tuned_folder + "/cosent-fine-tuned", 
        os.path.join(path_root,  "evals/data.cosent.npy")
    ], 
    "simbert": [
        fine_tuned_folder + "/simbert-fine-tuned", 
        os.path.join(path_root,  "evals/data.simbert.npy")
    ],
    'm3e-large': [
        fine_tuned_folder + "/m3e-fine-tuned", 
        os.path.join(path_root,  "evals/data.m3e.npy")
    ],
    'bge-large': [
        fine_tuned_folder + "/bge-fine-tuned", 
        os.path.join(path_root,  "evals/data.bge.npy")
    ]
}

# # dialogue编码模型
# dialog_model_path = "/chj/nsx/models/stella-dialogue-large-zh-v3-1792d"

# 粗排Prerank

# 精排rerank
rerank_cross_model = ["/bge-reranker-large", "/bge-reranker-v2-m3", "/bce-reranker-base_v1"]        # 分数倾向于两极分化
rerank_bi_model = ["/stella-mrl-large-zh-v3.5-1792d", "/Yinka", "/zpoint_large_embedding_zh"]       # 分数倾向于中间
rerank_model_path = model_cache_folder + rerank_cross_model[0]
use_cuda = True                             # 是否使用GPU
use_rank = True      

# IM对话数据
dialog_im_path = os.path.join(path_root, "data_factory/dialog_df_智能驾舱_20240101-20240226_4.csv")
dialog_im_processed_path = os.path.join(path_root, "data_factory/dialog_df_智能驾舱_processed.csv")

# 评测集 / 评测结果
testset_folder = os.path.join(path_root, 'evals/testset_new')
test_set_path = testset_folder + "/testset_update.csv"
# test_set_path = testset_folder + "/testset.xlsx"
eval_path = testset_folder + "/evaluate_{}_ts_{}.csv"
eval_path_filter = testset_folder + "/evaluate_{}_ts_{}_filter.csv"

# 训练数据
# train_similar_qq_path = os.path.join(path_root, "data_factory/processed_faq相似问_haixiao.csv")
# train_similar_qq_path = os.path.join(path_root, "data_factory/processed_crm_similar_pair_haixiao_0612.csv")
train_similar_qq_path = os.path.join(path_root, "data_factory/processed_crm_similar_pair.csv")

# 知识库
qa_pair_path = os.path.join(path_root, "data_factory/processed_crm_knowledge_pair.csv")
qa4api_path = os.path.join(path_root, "data_factory/qa4api.csv")
qa4rec_path = os.path.join(path_root, "data_factory/qa4rec.csv")

# 七鱼相关库
qa_qy_onetouch = kafka_data_folder + "/qa_qy_onetouch.csv"
qa_qy_slot_path = kafka_data_folder + "/slot.json"
qa_custom_greeting = os.path.join(path_root, "data_factory/qa_custom_greeting.csv")
qa_global_greeting = os.path.join(path_root, "data_factory/qa_global_greeting.csv")
qa_qy_slot_car_path = os.path.join(path_root, 'data_factory/entities/slot_car.json')
# kafka持久化存储
intent_path = kafka_data_folder + "/intent.json"
param_path = kafka_data_folder + "/param.json"
entity_path = kafka_data_folder + "/entity.json"
