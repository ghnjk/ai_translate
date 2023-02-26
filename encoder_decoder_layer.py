#!/usr/bin/env python3
# -*- coding:utf-8 _*-
"""
@file: encoder_decoder_layer
@author: jkguo
@create: 2023/2/13
"""
import tensorflow as tf
from attention import AdditiveAttentionLayer
import numpy as np
from transformer_layer import TransformerEncoderBlock, TransformerDecoderBlock


class PositionalEncoding(tf.keras.layers.Layer):

    def __init__(self, hidden_unit: int, dropout: float, max_len: int = 1000, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.dropout_layer = tf.keras.layers.Dropout(dropout)
        # 创建一个足够长的P
        # shape: 1, max_len, hidden_unit
        pos_values = np.zeros(shape=(1, max_len, hidden_unit), dtype=np.float32)
        # x shape: max_len, 1
        x = np.arange(max_len, dtype=np.float32).reshape(
            -1, 1
        )
        # x.shape: max_len, hidden_unit / 2
        x = x / np.power(10000, np.arange(0, hidden_unit, 2, dtype=np.float32) / hidden_unit)
        pos_values[:, :, 0::2] = tf.sin(x)
        pos_values[:, :, 1::2] = tf.cos(x)
        self.pos_values = tf.cast(pos_values, dtype=tf.float32)

    def call(self, x, *args, **kwargs):
        """
        :param x: shape: batch_size, step_len, hidden_unit
        :param args:
        :param kwargs:
        :return:
        """
        x = x + self.pos_values[:, :x.shape[1], :]
        return self.dropout_layer(x, **kwargs)


class EncoderLayer(tf.keras.layers.Layer):

    def __init__(self, enc_vocab_size: int, embedding_size: int, gru_layers: int, hidden_unit: int, dropout=0.0,
                 **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        self.enc_vocab_siz: int = enc_vocab_size
        self.embedding_size: int = embedding_size
        self.gru_layers: int = gru_layers
        self.hidden_unit: int = hidden_unit
        self.embedding = tf.keras.layers.Embedding(enc_vocab_size, embedding_size)
        self.gru_list = []
        for i in range(gru_layers):
            self.gru_list.append(
                tf.keras.layers.GRU(hidden_unit, go_backwards=True, return_state=True, return_sequences=True,
                                    dropout=dropout)
            )

    def call(self, inputs, enc_valid_len, *args, **kwargs):
        """
        :param inputs: shape: [batch_size, time_steps]
        :param enc_valid_len: shape: [batch_size]
        :param args:
        :param kwargs:
        :return:
        """
        training = kwargs["training"]
        x = self.embedding(inputs)
        states = []
        for gru in self.gru_list:
            x, state = gru(x, training=training)
            states.append(state)
        return x, states

    def get_config(self):
        config = super(EncoderLayer, self).get_config()
        config.update({
            "enc_vocab_size": self.enc_vocab_size,
            "embedding_size": self.embedding_size,
            "gru_layers": self.gru_layers,
            "hidden_unit": self.hidden_unit
        })
        return config


class TransformerEncoderLayer(tf.keras.layers.Layer):

    def __init__(self, enc_vocab_size: int, embedding_size: int, num_blocks: int, hidden_unit: int,
                 ffn_num_hidden: int, num_head: int, layer_norm_axis: tuple = 2, dropout=0.0, bias=False,
                 **kwargs):
        super(TransformerEncoderLayer, self).__init__(**kwargs)
        self.enc_vocab_siz: int = enc_vocab_size
        self.embedding_size: int = embedding_size
        self.num_blocks: int = num_blocks
        self.hidden_unit: int = hidden_unit
        self.embedding = tf.keras.layers.Embedding(enc_vocab_size, embedding_size)
        self.pos_embedding = PositionalEncoding(hidden_unit=self.hidden_unit, dropout=dropout)
        self.blocks = []
        for i in range(self.num_blocks):
            self.blocks.append(
                TransformerEncoderBlock(num_hidden=self.hidden_unit, layer_norm_axis=layer_norm_axis,
                                        ffn_num_hidden=ffn_num_hidden, num_head=num_head, dropout=dropout, bias=bias)
            )
        self.attention_weights = []

    def call(self, x, valid_lens, *args, **kwargs):
        """

        :param x: shape: batch_size, sequence_len, features
        :param valid_lens:  batch_size
        :param args:
        :param kwargs:
        :return:
        """
        # 因为位置编码值在-1和1之间，
        # 因此嵌入值乘以嵌入维度的平方根进行缩放，
        # 然后再与位置编码相加。
        x = self.pos_embedding(
            self.embedding(x) * tf.math.sqrt(tf.cast(self.hidden_unit, dtype=tf.float32)),
            **kwargs
        )
        self.attention_weights = []
        for blk in self.blocks:
            x = blk(x, valid_lens, **kwargs)
            self.attention_weights.append(
                blk.attention.attention.attention_weights
            )
        return x


class DecoderLayer(tf.keras.layers.Layer):

    def __init__(self, dec_vocab_size: int, embedding_size: int, gru_layers: int, hidden_unit: int, dropout=0.0,
                 **kwargs):
        super(DecoderLayer, self).__init__(**kwargs)
        self.dropout = dropout
        self.dec_vocab_size: int = dec_vocab_size
        self.embedding_size: int = embedding_size
        self.gru_layers: int = gru_layers
        self.hidden_unit: int = hidden_unit
        self.gru_list = []
        for i in range(gru_layers):
            self.gru_list.append(
                tf.keras.layers.GRU(hidden_unit, go_backwards=True, return_state=True, return_sequences=True,
                                    dropout=dropout)
            )
        self.embedding = tf.keras.layers.Embedding(dec_vocab_size, embedding_size)
        self.dense = tf.keras.layers.Dense(dec_vocab_size, activation="softmax")
        self.has_attention_weights = False

    def init_state(self, enc_all_outputs, enc_valid_len, *args):
        # 将encoder 最后的一层的最后state作为context
        state = enc_all_outputs[1][-1]
        context = tf.reshape(state, shape=(-1, 1, state.shape[1]))
        return enc_all_outputs[1], context

    def calc_context_state(self, inputs, states, context, training):
        time_steps = inputs.shape[1]
        context_state = tf.repeat(context, repeats=time_steps, axis=1)
        return context_state

    def call(self, inputs, *args, **kwargs):
        """
        :param inputs: shape [batch_size, time_steps]
        :param args:
        :param kwargs:
        :return:
        """
        states = kwargs["states"]
        context = kwargs['context']
        training = kwargs["training"]
        inputs = self.embedding(inputs)
        context_state = self.calc_context_state(inputs, states, context, training)
        decode_inputs_with_ctx = tf.concat([
            inputs, context_state
        ], axis=2)
        x = decode_inputs_with_ctx
        new_states = []
        for i, gru in enumerate(self.gru_list):
            x, state = gru(x, initial_state=states[i], training=training)
            new_states.append(state)
        return self.dense(x), new_states

    def get_config(self):
        config = super(DecoderLayer, self).get_config()
        config.update({
            "dec_vocab_size": self.dec_vocab_size,
            "embedding_size": self.embedding_size,
            "gru_layers": self.gru_layers,
            "hidden_unit": self.hidden_unit
        })
        return config


class AttentionDecoderLayer(DecoderLayer):

    def __init__(self, **kwargs):
        super(AttentionDecoderLayer, self).__init__(**kwargs)
        self.attention_layer = AdditiveAttentionLayer(self.hidden_unit, self.dropout)
        self._attention_weights = []
        self.has_attention_weights = True

    def init_state(self, enc_all_outputs, enc_valid_len, *args):
        enc_outputs, enc_states = enc_all_outputs
        context = (enc_outputs, enc_valid_len)
        return enc_states, context

    def calc_context_state(self, inputs, states, context, training):
        enc_outputs, enc_valid_len = context
        # 用当前最后一层的state作为查询条件，转shape batch_size, 1, hidden_unit
        # 查询个数固定为1
        queries = tf.expand_dims(states[-1], axis=1)
        # enc_outputs的形状为(batch_size,num_steps,num_hidden)
        # 用编码器的输出作为key和value， k-v count： num_steps, 特征数为 num_hidden
        keys = values = enc_outputs
        # 计算出来的attention shape： batch_size, query_count, value_features
        attention = self.attention_layer(queries, keys, values, enc_valid_len, training=training)
        return attention

    def call(self, inputs, *args, **kwargs):
        """
        :param inputs: shape [batch_size, time_steps]
        :param args:
        :param kwargs:
        :return:
        """
        states = kwargs["states"]
        context = kwargs['context']
        training = kwargs["training"]
        inputs = self.embedding(inputs)  # shape: batch_size, time_steps, embedding_size
        inputs = tf.transpose(inputs, perm=(1, 0, 2))  # shape: time_steps, batch_size, embedding_size
        outputs = []
        self._attention_weights = []
        # 针对每个time_step进行计算
        for x in inputs:
            # x shape: batch_size, embedding_size
            # context_state: batch_size, 1, num_hidden
            context_state = self.calc_context_state(inputs, states, context, training)
            self._attention_weights.append(self.attention_layer.attention_weights)
            # 在特征维度上连结 decode_inputs_with_ctx
            decode_inputs_with_ctx = tf.concat([
                context_state,
                tf.expand_dims(x, axis=1)
            ], axis=-1)
            # shape: batch_size, time_step=1, num_hidden + embedding_size
            gx = decode_inputs_with_ctx
            new_states = []
            for i, gru in enumerate(self.gru_list):
                gx, state = gru(gx, initial_state=states[i], training=training)
                new_states.append(state)
            states = new_states
            outputs.append(gx)
        # outputs shape: batch_size, time_steps, num_hidden
        outputs = tf.concat(outputs, axis=1)
        return self.dense(outputs), states

    @property
    def attention_weights(self):
        return self._attention_weights


class TransformerDecoderLayer(tf.keras.layers.Layer):

    def __init__(self, dec_vocab_size: int, embedding_size: int, num_blocks: int,
                 hidden_unit: int, ffn_num_hidden: int, num_head: int,
                 dropout=0.1, layer_norm_axis=2, bias=False,
                 **kwargs):
        super(TransformerDecoderLayer, self).__init__(**kwargs)
        self.dropout = dropout
        self.dec_vocab_size: int = dec_vocab_size
        self.embedding_size: int = embedding_size
        self.num_blocks: int = num_blocks
        self.hidden_unit: int = hidden_unit
        self.blocks = []
        for i in range(num_blocks):
            self.blocks.append(
                TransformerDecoderBlock(
                    layer_idx=i, num_hidden=hidden_unit,
                    layer_norm_axis=layer_norm_axis, ffn_num_hidden=ffn_num_hidden,
                    num_head=num_head, dropout=dropout, bias=bias
                )
            )
        self.embedding = tf.keras.layers.Embedding(dec_vocab_size, embedding_size)
        self.pos_embedding = PositionalEncoding(hidden_unit=self.hidden_unit, dropout=dropout)
        self.dense = tf.keras.layers.Dense(dec_vocab_size, activation="softmax")
        self.has_attention_weights = True
        self._attention_weights = None

    def init_state(self, enc_all_outputs, enc_valid_len, *args):
        """
        :param enc_all_outputs:
        :param enc_valid_len:
        :param args:
        :return:
            [ enc_outputs, enc_valid_len, [None] * self.num_layers ]
        """
        state = [
            enc_all_outputs, enc_valid_len, [None] * self.num_blocks
        ]
        return state, None

    def call(self, inputs, states, context, *args, **kwargs):
        x = self.embedding(inputs)
        x = self.pos_embedding(
            x * tf.math.sqrt(tf.cast(self.hidden_unit, dtype=tf.float32)),
            **kwargs)
        self._attention_weights = [[], []]
        for blk in self.blocks:
            x, state = blk(x, states, **kwargs)
            self._attention_weights[0].append(
                blk.attention_layer_1.attention.attention_weights
            )
            self._attention_weights[1].append(
                blk.attention_layer_2.attention.attention_weights
            )
        return self.dense(x), states

    @property
    def attention_weights(self):
        return self._attention_weights
