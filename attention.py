#!/usr/bin/env python3
# -*- coding:utf-8 _*-
"""
@file: attention
@author: jkguo
@create: 2023/2/20
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def plot_attention_map(attention_weight, dark_color=None, light_color=None):
    """

    :param attention_weight: shape: row, col
    :param dark_color:
    :param light_color:
    :return:
    """
    if dark_color is None:
        dark_color = [.8, 0.2, 0.2]
    if light_color is None:
        light_color = [1.0, 1.0, 1.0]
    dark_color = np.array(dark_color)
    light_color = np.array(light_color)
    attention_weight = (attention_weight + 1.0) / 2.0
    img = light_color - np.expand_dims(attention_weight, axis=2) * (light_color - dark_color)
    plt.imshow(img)


def sequence_mask(x, valid_lens, value):
    mask = tf.range(start=0, limit=x.shape[1], dtype=tf.float32)[None, :] < tf.cast(valid_lens, dtype=tf.float32)[:,
                                                                            None]
    if len(x.shape) == 3:
        return tf.where(tf.expand_dims(mask, axis=-1), x, value)
    else:
        return tf.where(mask, x, value)


def mask_softmax(x, valid_lens):
    # x[valid_len: ] = -1e6
    # softmax(x)
    x_shape = x.shape
    if valid_lens is None:
        return tf.nn.softmax(x, axis=-1)
    if len(valid_lens.shape) == 1:
        valid_lens = tf.repeat(valid_lens, repeats=x_shape[1])
    else:
        valid_lens = tf.reshape(valid_lens, shape=-1)
    x = tf.reshape(x, shape=(-1, x_shape[-1]))
    # 最后一轴上被掩蔽的元素使用一个非常大的负值替换，从而其softmax输出为0
    x = sequence_mask(x, valid_lens, -1e6)
    x = tf.reshape(x, x_shape)
    return tf.nn.softmax(x, axis=-1)


class AdditiveAttentionLayer(tf.keras.layers.Layer):

    def __init__(self, num_hidden, dropout, **kwargs):
        super(AdditiveAttentionLayer, self).__init__(**kwargs)
        self.qh_layer = tf.keras.layers.Dense(num_hidden, use_bias=False)
        self.kh_layer = tf.keras.layers.Dense(num_hidden, use_bias=False)
        self.v_layer = tf.keras.layers.Dense(1, use_bias=False)
        self.dropout_layer = tf.keras.layers.Dropout(dropout)
        self.attention_weights = None

    def call(self, queries, keys, values, valid_lens, **kwargs):
        """
        :param queries:  查询特征 shape： batch_size, query_count, query_features
        :param keys: 条件特征 shape: batch_size, k_v_count, key_feature
        :param values: 值特征 shape: batch_size, k_v_count, value_features
        :param valid_lens: 每个批次的key-value有效长度： shape: batch_size,
        :param kwargs: batch_size, 查询个数, value_features
        :return:
        """
        queries = self.qh_layer(queries)
        keys = self.kh_layer(keys)
        # 在维度扩展后，
        # queries的形状：(batch_size，查询的个数，1，num_hidden)
        # key的形状：(batch_size，1，“键－值”对的个数，num_hidden)
        # 使用广播方式进行求和
        # features shape:  batch_size, 查询的个数, 键－值”对的个数, num_hidden
        features = tf.add(
            tf.expand_dims(queries, axis=2),
            tf.expand_dims(keys, axis=1)
        )
        features = tf.nn.tanh(features)
        # self.v_layer(features) shape:  batch_size, 查询的个数, 键－值”对的个数, 1
        # squeeze 后 shape： batch_size, 查询的个数, 键－值”对的个数
        scores = tf.squeeze(self.v_layer(features), axis=-1)
        # attention_weights shape: batch_size, 查询个数，键－值”对的个数
        self.attention_weights = mask_softmax(scores, valid_lens)
        # 返回shape： batch_size, 查询个数, value_features
        return tf.matmul(self.dropout_layer(self.attention_weights, **kwargs), values)


class DotProductionAttention(tf.keras.layers.Layer):
    """
    缩放点积注意力
    """

    def __init__(self, dropout, **kwargs):
        super(DotProductionAttention, self).__init__(**kwargs)
        self.dropout_layer = tf.keras.layers.Dropout(dropout)
        self.attention_weights = None

    def call(self, queries, keys, values, valid_lens, *args, **kwargs):
        """
            queries, keys 特征维度必须相同
        :param queries: shape: batch_size, query_count, d
        :param keys: shape: batch_size, kv_count, d
        :param values: shape: batch_size, kv_count, value_feature
        :param valid_lens: shape: (batch_size，)或者(batch_size，查询的个数)
        :param args:
        :param kwargs:
        :return:
        """
        d = queries.shape[-1]
        # scores= q * kT / sqrt(d) shape: batch_size, query_count, kv_count
        scores = tf.matmul(queries, keys, transpose_b=True) / tf.math.sqrt(
            tf.cast(d, dtype=tf.float32)
        )
        self.attention_weights = mask_softmax(scores, valid_lens)
        # 返回： attention * value -> shape: batch_size, query_count, value_feature
        return tf.matmul(
            self.dropout_layer(self.attention_weights, **kwargs),
            values
        )


class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, hidden_unit: int, num_head: int, dropout: float, bias: bool = False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_head: int = num_head
        self.attention = DotProductionAttention(dropout)
        self.w_q_layer = tf.keras.layers.Dense(hidden_unit, use_bias=bias)
        self.w_k_layer = tf.keras.layers.Dense(hidden_unit, use_bias=bias)
        self.w_v_layer = tf.keras.layers.Dense(hidden_unit, use_bias=bias)
        self.w_o_layer = tf.keras.layers.Dense(hidden_unit, use_bias=bias)

    @staticmethod
    def transpose_qkv(x, num_heads: int):
        """
        对x的最后一位分成 num_heads部分，在batch_size这维堆加
        :param x: shape batch_size, query_count/kv_count, d
        :param num_heads: int
        :return: shape: batch_size * num_heads, query_count/kv_count, d / num_heads
        """
        # 先转成： batch_size, query_count/kv_count, num_heads, d / num_heads
        x = tf.reshape(x, shape=(x.shape[0], x.shape[1], num_heads, -1))
        # 将 num_head调转到 batch_size后一维
        x = tf.transpose(x, perm=(0, 2, 1, 3))
        # 合并batch_size和num_head
        return tf.reshape(x, shape=(-1, x.shape[2], x.shape[3]))

    @staticmethod
    def reverse_transpose_qkv(x, num_heads: int):
        """
        transpose_qkv你操作
        :param x:  batch_size * num_heads, query_count/kv_count, d / num_heads
        :param num_heads: int
        :return: batch_size, query_count/kv_count, d
        """
        # 将 batch_size * num_heads 拆出来 shape: batch_size, num_heads, query_count/kv_count, d / num_heads
        x = tf.reshape(x, shape=(-1, num_heads, x.shape[1], x.shape[2]))
        # 将num_heads维调到 倒数第二维 shape : batch_size, query_count/kv_count, num_heads,  d / num_heads
        x = tf.transpose(x, perm=(0, 2, 1, 3))
        # 合并最后两维: shape: batch_size, query_count/kv_count, d
        return tf.reshape(x, shape=(x.shape[0], x.shape[1], -1))

    def call(self, queries, keys, values, valid_lens, *args, **kwargs):
        """
            queries 和 keys的特征维需要都是： d
            d 必须是 num_heads 的倍数
        :param queries: shape: batch_size, query_count, d
        :param keys:  shape: batch_size, kv_count, d
        :param values: shape: batch_size, kv_count, value_features
        :param valid_lens: (batch_size，)或(batch_size，查询的个数)
        :param args:
        :param kwargs:
        :return: batch_size, query_count, hidden_unit
        """
        # shape: batch_size * num_head, query_count, d / num_head
        queries = self.transpose_qkv(queries, self.num_head)
        # shape: batch_size * num_head, kv_count, d / num_head
        keys = self.transpose_qkv(keys, self.num_head)
        # shape: batch_size * num_head, kv_count, value_feature / num_head
        values = self.transpose_qkv(values, self.num_head)
        if valid_lens is not None:
            # 由于 第一位已经变成 batch_size * num_head
            # 所以，这儿valid_len需要同步增加num_head
            valid_lens = tf.repeat(valid_lens, repeats=self.num_head, axis=0)
        # 经过缩放点积注意力后 shape： batch_size * num_head, query_count, value_feature / num_head
        out = self.attention(queries, keys, values, valid_lens, **kwargs)
        # 需要转成： batch_size, query_count, value_feature
        out = self.reverse_transpose_qkv(out, self.num_head)
        # 最后dense shape: batch_size, query_count, hidden_unit
        return self.w_o_layer(out)
