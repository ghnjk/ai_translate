#!/usr/bin/env python3
# -*- coding:utf-8 _*-
"""
@file: transformer_layer
@author: jkguo
@create: 2023/2/25
"""
import tensorflow as tf
from attention import MultiHeadAttention


class PositionWiseFFN(tf.keras.layers.Layer):
    """基于位置的前馈网络"""

    def __init__(self, ffn_num_hidden, ffn_num_outputs, **kwargs):
        super().__init__(*kwargs)
        self.dense1 = tf.keras.layers.Dense(ffn_num_hidden)
        self.relu = tf.keras.layers.ReLU()
        self.dense2 = tf.keras.layers.Dense(ffn_num_outputs)

    def call(self, x, **kwargs):
        return self.dense2(self.relu(self.dense1(x)))


class AddNorm(tf.keras.layers.Layer):
    """残差连接后进行层规范化"""

    def __init__(self, layer_norm_axis, dropout, **kwargs):
        super().__init__(**kwargs)
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.ln = tf.keras.layers.LayerNormalization(layer_norm_axis)

    def call(self, x, y, **kwargs):
        return self.ln(self.dropout(y, **kwargs) + x)


class TransformerEncoderBlock(tf.keras.layers.Layer):

    def __init__(self, num_hidden: int, layer_norm_axis: tuple, ffn_num_hidden: int, num_head: int, dropout: float,
                 bias: bool = False, **kwargs):
        super(TransformerEncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(hidden_unit=num_hidden, num_head=num_head, dropout=dropout, bias=bias)
        self.add_norm_layer_1 = AddNorm(layer_norm_axis=layer_norm_axis, dropout=dropout)
        self.ffn_layer = PositionWiseFFN(ffn_num_hidden=ffn_num_hidden, ffn_num_outputs=num_hidden)
        self.add_norm_layer_2 = AddNorm(layer_norm_axis=layer_norm_axis, dropout=dropout)

    def call(self, x, valid_lens, *args, **kwargs):
        """
        :param x: shape: batch_size, sequence_len, features
        :param valid_lens:  batch_size
        :param args:
        :param kwargs:
        :return:
        """
        # shape = batch_size, sequence_len, num_hidden
        y = self.attention(x, x, x, valid_lens, **kwargs)
        y = self.add_norm_layer_1(x, y, **kwargs)
        # shape: batch_size, sequence_len, num_hidden
        ffn_y = self.ffn_layer(y)
        return self.add_norm_layer_2(y, ffn_y, **kwargs)


class TransformerDecoderBlock(tf.keras.layers.Layer):

    def __init__(self, layer_idx: int, num_hidden: int, layer_norm_axis, ffn_num_hidden: int, num_head: int,
                 dropout: float,
                 bias: bool = False, **kwargs):
        super(TransformerDecoderBlock, self).__init__(**kwargs)
        self.layer_idx: int = layer_idx
        self.attention_layer_1 = MultiHeadAttention(hidden_unit=num_hidden, num_head=num_head, dropout=dropout,
                                                    bias=bias)
        self.add_norm_layer_1 = AddNorm(layer_norm_axis=layer_norm_axis, dropout=dropout)
        self.attention_layer_2 = MultiHeadAttention(hidden_unit=num_hidden, num_head=num_head, dropout=dropout,
                                                    bias=bias)
        self.add_norm_layer_2 = AddNorm(layer_norm_axis=layer_norm_axis, dropout=dropout)
        self.ffn_layer = PositionWiseFFN(ffn_num_hidden=ffn_num_hidden, ffn_num_outputs=num_hidden)
        self.add_norm_layer_3 = AddNorm(layer_norm_axis=layer_norm_axis, dropout=dropout)

    def call(self, x, state, **kwargs):
        """
        :param x:
        :param state: [ enc_outputs, enc_valid_len, all_pre_predict_outputs ]
        :param kwargs:
        :return:
        """
        training = kwargs["training"]
        enc_outputs, enc_valid_len = state[0], state[1]
        # 训练阶段，输出序列的所有词元都在同一时间处理，
        # 因此state[2][self.layer_idx]初始化为None。
        # 预测阶段，输出序列是通过词元一个接着一个解码的，
        # 因此state[2][self.layer_idx]包含着一开始直到当前时间步第i个块解码的输出表示
        if training:
            key_values = x
            batch_size, num_steps, _ = x.shape
            # dec_valid_lens的开头:(batch_size,num_steps),
            # 其中每一行是[1,2,...,num_steps]
            # 用来在训练模式时， 针对每个step的查询，掩盖掉后续的信息
            dec_valid_lens = tf.repeat(
                tf.reshape(tf.range(1, num_steps + 1), shape=(-1, num_steps)),
                repeats=batch_size,
                axis=0
            )
        else:
            if state[2][self.layer_idx] is None:
                key_values = x
            else:
                key_values = tf.concat((state[2][self.layer_idx], x), axis=1)
            # 这儿将当前层的之前的输入和当前x合并后保存到state[2][self.layer_idx]
            state[2][self.layer_idx] = key_values
            # 预测模式时， 每个单词进行预测，看不到后续的信息，无需掩盖
            dec_valid_lens = None
        # 自注意力
        x1 = self.attention_layer_1(x, key_values, key_values, dec_valid_lens, **kwargs)
        y1 = self.add_norm_layer_1(x, x1, **kwargs)
        # 编码器－解码器注意力
        y2 = self.attention_layer_2(y1, enc_outputs, enc_outputs, enc_valid_len, **kwargs)
        y3 = self.add_norm_layer_2(y1, y2, **kwargs)
        z = self.ffn_layer(y3, **kwargs)
        return self.add_norm_layer_3(y3, z, **kwargs), state
