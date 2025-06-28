#! -*- coding: utf-8 -*-

import keras.backend as K
import tensorflow as tf


# 普通sparse交叉熵，以logits为输入
def sparse_logits_categorical_crossentropy(y_true, y_pred, scale=30):
    return K.sparse_categorical_crossentropy(y_true, scale * y_pred, from_logits=True)


def sparse_categorical_crossentropy(y_true, y_pred):
    """自定义稀疏交叉熵
    这主要是因为keras自带的sparse_categorical_crossentropy不支持求二阶梯度。
    """
    y_true = K.reshape(y_true, K.shape(y_pred)[:-1])
    y_true = K.cast(y_true, "int32")
    y_true = K.one_hot(y_true, K.shape(y_pred)[-1])
    return K.categorical_crossentropy(y_true, y_pred)


# one-hot版AM-Softmax
def amsoftmax_loss(y_true, y_pred, scale=30, margin=0.35):
    y_pred = y_true * (y_pred - margin) + (1 - y_true) * y_pred
    y_pred *= scale
    return K.categorical_crossentropy(y_true, y_pred, from_logits=True)


# 稀疏版AM-Softmax
def sparse_amsoftmax_loss(y_true, y_pred, scale=30, margin=0.45):
    """
    y_pred=np.array([[0.3, 0.8, 0.9], [0.2,0.7,0.1]])
    y_true=np.array([[0], [1]])
    batch_idxs = [[0], [1]]
    idxs = [[0, 0], [1, 1]]
    y_true_pred = [[0.3], [0.7]]
    y_true_pred_margin = [[0.3-m], [0.7-m]]
    _Z = s * [[0.3, 0.8, 0.9, 0.3-m], [0.2,0.7,0.1, 0.7-m]]
    logZ = [[27.04858737],[21.04858766]]
    logZ = [[27.04858735],[18.00000645]]
    return [[2.10485874e+01],[6.45009388e-06]]
    """
    y_true = K.expand_dims(y_true[:, 0], 1) # 保证y_true的shape=(None, 1)
    y_true = K.cast(y_true, 'int32') # 保证y_true的dtype=int32
    batch_idxs = K.arange(0, K.shape(y_true)[0])
    batch_idxs = K.expand_dims(batch_idxs, 1)
    idxs = K.concatenate([batch_idxs, y_true], 1)
    y_true_pred = tf.gather_nd(y_pred, idxs) # 目标特征，用tf.gather_nd提取出来
    y_true_pred = K.expand_dims(y_true_pred, 1)
    y_true_pred_margin = y_true_pred - margin # 减去margin 
    _Z = K.concatenate([y_pred, y_true_pred_margin], 1) # 为计算配分函数
    _Z = _Z * scale # 缩放结果，主要因为pred是cos值，范围[-1, 1]
    logZ = K.logsumexp(_Z, 1, keepdims=True) # 用logsumexp，保证梯度不消失  
    logZ = logZ + K.log(1 - K.exp(scale * y_true_pred - logZ)) # 从Z中减去exp(scale * y_true_pred)
    return - y_true_pred_margin * scale + logZ


# 简单的类A-Softmax（m=4）
def sparse_simpler_asoftmax_loss(y_true, y_pred, scale=30):
    y_true = K.expand_dims(y_true[:, 0], 1) # 保证y_true的shape=(None, 1)
    y_true = K.cast(y_true, 'int32') # 保证y_true的dtype=int32
    batch_idxs = K.arange(0, K.shape(y_true)[0])
    batch_idxs = K.expand_dims(batch_idxs, 1)
    idxs = K.concatenate([batch_idxs, y_true], 1)
    y_true_pred = K.tf.gather_nd(y_pred, idxs) # 目标特征，用tf.gather_nd提取出来
    y_true_pred = K.expand_dims(y_true_pred, 1)
    # 用到了四倍角公式进行展开
    y_true_pred_margin = 1 - 8 * K.square(y_true_pred) + 8 * K.square(K.square(y_true_pred))
    # 下面等效于min(y_true_pred, y_true_pred_margin)
    y_true_pred_margin = y_true_pred_margin - K.relu(y_true_pred_margin - y_true_pred)
    _Z = K.concatenate([y_pred, y_true_pred_margin], 1) # 为计算配分函数
    _Z = _Z * scale # 缩放结果，主要因为pred是cos值，范围[-1, 1]
    logZ = K.logsumexp(_Z, 1, keepdims=True) # 用logsumexp，保证梯度不消失
    logZ = logZ + K.log(1 - K.exp(scale * y_true_pred - logZ)) # 从Z中减去exp(scale * y_true_pred)
    return - y_true_pred_margin * scale + logZ
