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
    if dark_color is None:
        dark_color = [.8, 0.2, 0.2]
    if light_color is None:
        light_color = [1.0, 1.0, 1.0]
    dark_color = np.array(dark_color)
    light_color = np.array(light_color)
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
