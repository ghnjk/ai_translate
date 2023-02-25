#!/usr/bin/env python3
# -*- coding:utf-8 _*-
"""
@file: encoder_decoder_layer
@author: jkguo
@create: 2023/2/13
"""
import tensorflow as tf
from attention import AdditiveAttentionLayer


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

    def call(self, inputs, *args, **kwargs):
        """
        :param inputs: shape: [batch_size, time_steps]
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
