import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import re
import random
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf

import keras
from keras.backend import tensorflow_backend
from keras.callbacks import TensorBoard
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda
from keras.layers import Conv2D, GlobalAveragePooling2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import RMSprop, SGD, Nadam
from keras.regularizers import l2
from keras.constraints import unit_norm
from keras import backend as K

from bert4keras.models import build_transformer_model as build_bert_model
# from bert4keras.bert import build_bert_model
# from bert4keras.layers import LayerNormalization
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import (
    Adam,
    extend_with_weight_decay,
    extend_with_exponential_moving_average,
)
from bert4keras.optimizers import *
from bert4keras.backend import keras, set_gelu

import util
from training.bertModel.margin_softmax import sparse_amsoftmax_loss

"""
参考
https://blog.csdn.net/nima1994/article/details/83862502
https://www.pyimagesearch.com/2020/11/30/siamese-networks-with-keras-tensorflow-and-deep-learning/#pyis-cta-modal
https://www.pyimagesearch.com/2021/01/18/contrastive-loss-for-siamese-networks-with-keras-and-tensorflow/
https://aistudio.baidu.com/aistudio/projectdetail/2051331?channelType=0&channel=0
https://kexue.fm/archives/7094
https://github.com/bojone/margin-softmax/blob/master/sent_sim.py
"""

# sets random seed
seed = 123
random.seed(seed)
np.random.seed(seed)

# set GPU memory
# 方法1:显存占用会随着epoch的增长而增长,也就是后面的epoch会去申请新的显存,前面已完成的并不会释放,为了防止碎片化
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 按需求增长
sess = tf.Session(config=config)
K.set_session(sess)

# specify the batch size and number of epochs
LR = 2e-5  # [3e-4, 5e-5, 2e-5] 默认学习率是0.001
SGD_LR = 0.001
warmup_proportion = 0.1  # 学习率预热比例
weight_decay = 0.01  # 权重衰减系数，类似模型正则项策略，避免模型过拟合
DROPOUT_RATE = 0.3  # 0.1
BATCH_SIZE = 32  # 16
EPOCHS = 25
maxlen = 64  # 最大不能超过512, 若出现显存不足，请适当调低这一参数
EB_SIZE = 128
scale, margin = 30, 0.15  # amsoftmax参数 30, 0.35
optimizer = "adam"  # "sgd" "adamW" "adamwlr":带权重衰减和warmup的优化器
kid2label, label2kid = {}, {}  # kid转换成递增的id格式


def custom_amsoftmax_loss(scale, margin):
    # 采用了闭包的方式，将参数传给 sparse_amsoftmax_loss，再调用 inner
    def inner(y_true, y_pred):
        return sparse_amsoftmax_loss(y_true,y_pred,scale, margin)
    return inner


class AmsoftmaxNluRecallModel:
    def __init__(self, cls_num, root):
        # 优化器选择
        self.opt_dict = {
            "sgd": SGD(LR, decay=1e-5, momentum=0.9, nesterov=True),
            "adam": Adam(LR, clipvalue=1.0),
            "nadam": Nadam(LR, clipvalue=1.0),
            "rmsprop": RMSprop(LR, clipvalue=1.0),
            "adamw": extend_with_weight_decay(Adam, "AdamW")(LR, weight_decay_rate=weight_decay),
            "adamlr": extend_with_piecewise_linear_lr(Adam, "AdamLR")(learning_rate=LR, lr_schedule={1000: 1.0}),
            "adamga": extend_with_gradient_accumulation(Adam, "AdamGA")(learning_rate=LR, grad_accum_steps=10),
            "adamla": extend_with_lookahead(Adam, "AdamLA")(learning_rate=LR, steps_per_slow_update=5, slow_step_size=0.5),
            "adamlo": extend_with_lazy_optimization(Adam, "AdamLO")(learning_rate=LR, include_in_lazy_optimization=[]),
            "adamwlr":extend_with_piecewise_linear_lr(extend_with_weight_decay(Adam, "AdamW"), "AdamWLR")(
                learning_rate=LR, weight_decay_rate=0.01, lr_schedule={1000: 1.0}
            ),
            "adamema": extend_with_exponential_moving_average(Adam, name="AdamEMA")(LR, ema_momentum=0.9999)
        }
        self.cls_num = int(cls_num)
        self.root = root
        self.build_model(self.root)
        self.get_tokenizer(self.root)

    def get_tokenizer(self, root, pre_tokenize=None):
        """建立分词器
        """
        self.tokenizer = Tokenizer(os.path.join(root, "vocab.txt"), do_lower_case=True, pre_tokenize=pre_tokenize)

    def get_bert_model(self, root):
        # bert模型不可被封装！！
        bert_model = build_bert_model(
            config_path=os.path.join(root, "bert_config.json"),
            checkpoint_path=os.path.join(root, "bert_model.ckpt"),
            # with_pool='linear',
            # return_keras_model=False,
            # model="bert",
        )
        # # Freeze the BERT model to reuse the pretrained features without modifying them.
        # for l in bert_model.layers:
        #     l.trainable = False
        return bert_model

    def build_model(self):
        x1_in = Input(shape=(None,))
        x2_in = Input(shape=(None,))
        # r_out = Input(shape=(None,), dtype="int32")

        bert_model = self.get_bert_model(self.root)

        x = bert_model([x1_in, x2_in])
        first_token = Lambda(lambda x: x[:, 0])(x)
        embedding = first_token
        first_token = Dropout(DROPOUT_RATE, name="dp1")(first_token)  #防止过拟合
        first_token = Lambda(lambda v: K.l2_normalize(v, 1))(first_token)  # 特征归一化（l2正则）√
        # embedding = first_token     # 推理阶段→dot

        first_token = Dropout(DROPOUT_RATE, name="dp1")(first_token)   #防止过拟合
        # first_token = Batchnormalization()(first_token)

        first_out = Dense(
            self.cls_num,
            name="dense_output",
            use_bias=False,  # no bias √
            kernel_constraint=unit_norm(),    # 权重归一化（单位范数（unit_form），限制权值大小为 1.0）√
            kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02),
        )(first_token)
        p_out = first_out

        self.encoder = Model([x1_in, x2_in], embedding) # 最终的目的是要得到一个编码器
        self.train_model = Model([x1_in, x2_in], p_out) # 用分类问题做训练
        # self.encoder = Model([x1_in, x2_in], embedding) # 最终的目的是要得到一个编码器
        # self.train_model = Model([x1_in, x2_in, r_out], p_out) # 用分类问题做训练

        # final_loss = sparse_amsoftmax_loss(r_out, p_out, scale, margin)
        # self.train_model.add_loss(final_loss)
        self.train_model.summary()


def seq_padding(ML):
    """将序列padding到同一长度, value=0, mode='post'"""

    def func(X, padding=0):
        return np.array(
            [
                np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x
                for x in X
            ]
        )

    return func


def compute_pair_input_arrays(input_arrays, maxlen, tokenizer):
    inp1_1, inp1_2 = [], []
    for instance in input_arrays:
        x1, x2 = tokenizer.encode(instance, maxlen=maxlen)
        inp1_1.append(x1)
        inp1_2.append(x2)

    L = [len(x) for x in inp1_1]
    ML = max(L) if L else 0

    pad_func = seq_padding(ML)
    res = [inp1_1, inp1_2]
    res = list(map(pad_func, res))
    return res


# def predict_emb(sent_array, nn_model):
#     X1 = []
#     X2 = []
#     for sent in sent_array:
#         text = sent[:maxlen]
#         x1, x2 = tokenizer.encode(text)
#         X1.append(x1)
#         X2.append(x2)
#     # X1 = [x1]
#     # X2 = [x2]
#     X1 = seq_padding(X1)
#     X2 = seq_padding(X2)
#     _, emb = nn_model.predict([X1, X2], batch_size=200)
#     # print(np.array(emb).shape)
#     return emb


if __name__ == "__main__":

    # import sys
    # sys.path.append(r"../")
    from configs.config import *

    gen_train_file = "train_data/train.csv"
    model_path = "model/nlu_sort_amsoftmax.weights.best.h5"
    checkpoint_path = "model/nlu_sort_amsoftmax-{epoch:02d}.ckpt"    # 模型保存成 ckpt 格式
    root = [
        "../corpus/chinese_wwm_ext_L-12_H-768_A-12",    # 如果选择哈工大中文模型，则设置LR=5e-5
        "../corpus/chinese_simbert_L-12_H-768_A-12",    # SimBERT
    ]
    cls_num = 5389

    print("[INFO] building model...")
    arm = AmsoftmaxNluRecallModel(cls_num, root[1])

    df = pd.read_csv(
        similar_q_zhipu_path, header=0, sep=",", encoding="utf-8", engine="python"
    )
    df = df.applymap(lambda x: re.sub(r"[\"”“]", "", str(x)))
    df = df.dropna()
    grouped = df.groupby("standard_question")
    # for name, group in grouped:
    #     kid2label.setdefault(name, len(kid2label))
    # label2kid = {v: k for k, v in kid2label.items()}

    x_list, y_list = [], []
    for name, group in grouped:
        for i,row in group.iterrows():
            if row["standard_question"] not in x_list:
                x_list.extend([row["standard_question"], row["similar_question"]])
                y_list.extend([i] *2)
            else:
                x_list.append(row['res'])
                y_list.extend([i])

    x_train, y_train = (
        np.array(x_list),
        np.array(y_list),
        # np.array(list(map(lambda x: kid2label[x], y_list))),
    )

    print("padding前训练集大小：", x_train.shape, y_train.shape)  # (172836,) (172836,)
    x_train = compute_pair_input_arrays(x_train, maxlen, tokenizer=arm.tokenizer)
    print("padding后训练集大小：", x_train[0].shape, x_train[1].shape)  # (71480, 64) (71480, 64)
    print("\n")
    train_inputs = []
    train_inputs.extend(x_train)
    train_inputs.append(y_train)

    # compile the model
    print("[INFO] compiling model...")
    arm.train_model.compile(
        loss=custom_amsoftmax_loss(scale, margin),
        optimizer=arm.opt_dict['adam'], 
        metrics=["sparse_categorical_accuracy"]  # "acc"
    )  # 用足够小的学习率

    # custom_callback = CustomCallback(
    #     # valid_data=valid_data,  # (input, [kid])
    #     # test_data=test_data,  # (primary, kid)
    #     batch_size=BATCH_SIZE,
    #     encoder=encoder,
    # )
    # early_stopping = EarlyStopping(monitor='loss', patience=3)
    # # # 保存模型参数 ckpt 格式
    # # cp_callback = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

    # train the model
    print("[INFO] training model...")
    history = arm.train_model.fit(
        # train_inputs,
        x=x_train, 
        y=y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        shuffle=True,
        # callbacks=[custom_callback],
        # callbacks=[custom_callback, cp_callback],
    )


    # ========================================================== #


    # # 加载权重,测试相似度
    # encoder.load_weights(model_path)
    # test_x = compute_pair_input_arrays(["成本管理平台提示账号禁用"], maxlen, tokenizer=tokenizer)
    # test_query_vec = encoder.predict(
    #     test_x, batch_size=BATCH_SIZE*10
    # )
    # all_cand_text_ids = compute_pair_input_arrays(["成本管理平台无法新增合同", "成本管理平台合作方登陆系统显示账号被禁用"], maxlen, tokenizer=tokenizer)
    # all_cand_vecs = encoder.predict(
    #     all_cand_text_ids, batch_size=BATCH_SIZE*10
    # )
    # # dot_list = np.dot(all_cand_vecs, test_query_vec[0])
    # dot_list = cosine_similarity(all_cand_vecs, test_query_vec)
    # dot_list = [x[0] for x in dot_list]
    # print(dot_list)

    # # tSNE降维可视化
    # encoder.load_weights(model_path)
    # x_train = [x_train[0][100], x_train[1][:100]]
    # y = y_train[:100]
    # X  = encoder.predict(x_train, batch_size=BATCH_SIZE*10, verbose=1)
    # from sklearn import manifold, datasets
    # X_tsne = manifold.TSNE(n_components=2, init='random', random_state=5, verbose=1).fit_transform(X)
    # # Data Visualization
    # x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    # X_norm = (X_tsne-x_min) / (x_max-x_min)  #Normalize
    # plt.figure(figsize=(8, 8))
    # for i in range(X_norm.shape[0]):
    #     plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]), 
    #             fontdict={'weight': 'bold', 'size': 9})
    # plt.xticks([])
    # plt.yticks([])
    # plt.savefig("tsne.png")

    # Kmeans聚类→计算NMI、AMI指标

    # # SimCSE训练&评测
    # import sys
    # sys.path.append(r"./SimCSE")
    # from eval import train_cse, CustomCallback
    # encoder = Model(inputs=model.input, outputs=model.get_layer('dp1').output)
    # train_cse(encoder)
    # # 评测
    # custom_callback = CustomCallback(
    #     # valid_data=valid_data,  # (input, [kid])
    #     # test_data=test_data,  # (primary, kid)
    #     batch_size=32,
    #     encoder=encoder,
    # )
    # custom_callback.on_epoch_end(epoch=1)