#!/usr/bin/env python3
# -*- coding:utf-8 _*-
"""
@file: translate_model
@author: jkguo
@create: 2023/2/24
"""
import sys
import math
import datetime
import typing
import numpy as np
import tensorflow as tf
import progressbar
from vocab import Vocab, BEGIN_SENTENCE, END_SENTENCE
from data_generator import TranslateDataGenerator
from encoder_decoder_layer import EncoderLayer, DecoderLayer, AttentionDecoderLayer, TransformerEncoderLayer, \
    TransformerDecoderLayer


def sequence_mask(x, valid_len, value=0):
    """在序列中屏蔽不相关的项"""
    max_len = x.shape[1]
    mask = tf.range(start=0, limit=max_len, dtype=tf.float32)[
           None, :] < tf.cast(valid_len[:, None], dtype=tf.float32)

    if len(x.shape) == 3:
        return tf.where(tf.expand_dims(mask, axis=-1), x, value)
    else:
        return tf.where(mask, x, value)


class MaskedSparseCategoricalCrossEntropy(tf.keras.losses.Loss):

    def __init__(self, valid_len: int):
        super().__init__(reduction='none')
        self.valid_len = valid_len

    def call(self, y_true, y_pred):
        """

        :param y_true: shape [batch_size, time_steps]
        :param y_pred: shape [batch_size, time_steps, feature_count]
        :return:
        """
        weights = tf.ones_like(y_true, dtype=tf.float32)
        weights = sequence_mask(weights, self.valid_len)
        label_one_hot = tf.one_hot(y_true, depth=y_pred.shape[-1])
        unweighted_loss = tf.keras.losses.CategoricalCrossentropy(reduction='none')(label_one_hot, y_pred)
        # Loss function should always return a vector of length batch_size. Because you have to return a loss for
        # each datapoint. Ex - If you are fitting data with a batch size of 32, then you need to return a vector of
        # length 32 from your loss function.
        weighted_loss = tf.reduce_mean((unweighted_loss * weights), axis=1)
        return weighted_loss


def grad_clipping(grads, theta):  # @save
    """Clip the gradient."""
    theta = tf.constant(theta, dtype=tf.float32)
    new_grad = []
    for grad in grads:
        if isinstance(grad, tf.IndexedSlices):
            new_grad.append(tf.convert_to_tensor(grad))
        else:
            new_grad.append(grad)
    norm = tf.math.sqrt(sum((tf.reduce_sum(grad ** 2)).numpy()
                            for grad in new_grad))
    norm = tf.cast(norm, tf.float32)
    if tf.greater(norm, theta):
        for i, grad in enumerate(new_grad):
            new_grad[i] = grad * theta / norm
    else:
        new_grad = new_grad
    return new_grad


def bleu_acc(y_true: typing.List[str], y_pred: typing.List[str], k: int):
    """
    用BLEU算法评测预测序列的准确性
    :param y_true:
    :param y_pred:
    :param k:
    :return:
    """
    score = math.exp(
        min(0.0, 1.0 - len(y_true) * 1.0 / len(y_pred))
    )
    for n in range(1, min(len(y_pred), k + 1)):
        mp = {}
        for s in range(len(y_true) - n + 1):
            key = '<--->'.join(y_true[s: s + n])
            if key in mp:
                mp[key] += 1
            else:
                mp[key] = 1
        all_count = 0
        match_count = 0
        for s in range(len(y_pred) - n + 1):
            key = '<--->'.join(y_pred[s: s + n])
            all_count += 1
            if mp.get(key, 0) > 0:
                match_count += 1
                mp[key] -= 1
        score *= math.pow(
            match_count * 1.0 / all_count,
            math.pow(0.5, n)
        )
    return score


class TranslateModelBase(tf.keras.Model):

    def __init__(self, **kwargs):
        super(TranslateModelBase, self).__init__(**kwargs)
        self.encoder: EncoderLayer = self.build_encoder_layer()
        self.decoder: DecoderLayer = self.build_decoder_layer()

    def build_encoder_layer(self) -> EncoderLayer:
        raise NotImplementedError(
            "build_encoder_layer must implements"
        )

    def build_decoder_layer(self) -> DecoderLayer:
        raise NotImplementedError(
            "build_decoder_layer must implements"
        )

    def call(self, inputs, training=None, mask=None):
        enc_x, enc_valid_len, dec_x = inputs
        enc_all_outputs = self.encoder(enc_x, enc_valid_len, training=training)
        enc_states, context = self.decoder.init_state(enc_all_outputs, enc_valid_len)
        dec_outputs, dec_states = self.decoder(dec_x, states=enc_states, context=context, training=training)
        return dec_outputs

    def __generate_tensor_graph(self, data_gen: TranslateDataGenerator,
                                train_summary_writer, train_log_dir, optimizer):
        tf.summary.trace_on(graph=True, profiler=False)
        self.__train_step(data_gen, 0, 32, optimizer)
        with train_summary_writer.as_default():
            tf.summary.trace_export(
                name="graph",  # <- Name of tag
                step=0,
                profiler_outdir=train_log_dir)

    def __train_step(self, data_gen, start, end, optimizer):
        sub_enc_x, sub_enc_valid_len, sub_dec_x, sub_target_valid_len, sub_target_y = data_gen[start: end]
        with tf.GradientTape() as tape:
            y_pred = self.call((sub_enc_x, sub_enc_valid_len, sub_dec_x), training=True)
            loss_fn = MaskedSparseCategoricalCrossEntropy(sub_target_valid_len)
            loss = loss_fn(sub_target_y, y_pred)
        gradients = tape.gradient(loss, self.trainable_variables)
        gradients = grad_clipping(gradients, 1.0)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss

    def custom_fit(
            self,
            data_gen: TranslateDataGenerator,
            batch_size=32,
            epochs=1,
            learning_rate=1e-3
    ):
        x_len = len(data_gen)
        step_count = math.ceil(len(data_gen) / batch_size)
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate)

        train_loss_metric = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/' + current_time + '/train'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        self.__generate_tensor_graph(data_gen, train_summary_writer, train_log_dir, optimizer)
        for epoch in range(epochs):
            si_list = np.arange(step_count)
            np.random.shuffle(si_list)
            print(f"Epoch {epoch + 1}/{epochs}")
            bar = progressbar.ProgressBar(maxval=step_count,
                                          widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()],
                                          fd=sys.stdout)
            bar.start()
            for step, si in enumerate(si_list):
                start = si * batch_size
                end = min(start + batch_size, x_len)
                loss = self.__train_step(data_gen, start, end, optimizer)
                train_loss_metric(loss)
                bar.update(step)
            bar.finish()
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss_metric.result(), step=epoch)
            template = 'Epoch {}, Loss: {:.4f}'
            print(template.format(epoch + 1,
                                  train_loss_metric.result()))
            train_loss_metric.reset_states()

    def translate(
            self,
            enc_x,
            enc_valid_len,
            dec_vocab: Vocab,
            max_len: int = 50
    ):
        enc_all_outputs = self.encoder(enc_x, enc_valid_len, training=False)
        enc_states, context = self.decoder.init_state(enc_all_outputs, enc_valid_len)
        dec_x = np.array([
            dec_vocab[BEGIN_SENTENCE]
        ]).reshape((1, 1))
        outputs = []
        dec_states = enc_states
        attention_weights = []
        for i in range(max_len):
            dec_outputs, dec_states = self.decoder(dec_x, states=dec_states, context=context, training=False)
            if self.decoder.has_attention_weights:
                attention_weights.append(
                    self.decoder.attention_weights
                )
            dec_outputs = tf.argmax(dec_outputs, axis=2)
            pred = dec_outputs[0][0]
            pred = dec_vocab.to_tokens(pred)
            if pred == END_SENTENCE:
                break
            outputs.append(pred)
            dec_x = dec_outputs
        if dec_vocab.token_mode == "word":
            return ' '.join(outputs), attention_weights
        else:
            return ''.join(outputs), attention_weights


class Seq2SeqTranslateModel(TranslateModelBase):

    def __init__(self, enc_vocab_size: int, enc_embedding_size, dec_vocab_size, dec_embedding_size,
                 dropout=0.0, gru_layers: int = 3, gru_hidden_units: int = 32, name="translate_model", **kwargs):
        self.enc_vocab_size: int = enc_vocab_size
        self.dec_vocab_size: int = dec_vocab_size
        self.gru_layers: int = gru_layers
        self.dropout = dropout
        self.gru_hidden_units: int = gru_hidden_units
        self.dec_embedding_size: int = dec_embedding_size
        self.enc_embedding_size: int = enc_embedding_size
        super(Seq2SeqTranslateModel, self).__init__(name=name, **kwargs)

    def build_encoder_layer(self) -> EncoderLayer:
        return EncoderLayer(enc_vocab_size=self.enc_vocab_size,
                            embedding_size=self.enc_embedding_size, dropout=self.dropout,
                            gru_layers=self.gru_layers, hidden_unit=self.gru_hidden_units, name="gru_encoder")

    def build_decoder_layer(self) -> DecoderLayer:
        return DecoderLayer(dec_vocab_size=self.dec_vocab_size,
                            embedding_size=self.dec_embedding_size, dropout=self.dropout,
                            gru_layers=self.gru_layers, hidden_unit=self.gru_hidden_units, name="gru_decoder")


class AttentionSeq2SeqTranslateModel(TranslateModelBase):

    def __init__(self, enc_vocab_size: int, enc_embedding_size, dec_vocab_size, dec_embedding_size,
                 dropout=0.0, gru_layers: int = 3, gru_hidden_units: int = 32, name="translate_model", **kwargs):
        self.enc_vocab_size: int = enc_vocab_size
        self.dec_vocab_size: int = dec_vocab_size
        self.gru_layers: int = gru_layers
        self.dropout = dropout
        self.gru_hidden_units: int = gru_hidden_units
        self.dec_embedding_size: int = dec_embedding_size
        self.enc_embedding_size: int = enc_embedding_size
        super(AttentionSeq2SeqTranslateModel, self).__init__(name=name, **kwargs)

    def build_encoder_layer(self) -> EncoderLayer:
        return EncoderLayer(enc_vocab_size=self.enc_vocab_size,
                            embedding_size=self.enc_embedding_size, dropout=self.dropout,
                            gru_layers=self.gru_layers, hidden_unit=self.gru_hidden_units, name="gru_encoder")

    def build_decoder_layer(self) -> DecoderLayer:
        return AttentionDecoderLayer(dec_vocab_size=self.dec_vocab_size,
                                     embedding_size=self.dec_embedding_size, dropout=self.dropout,
                                     gru_layers=self.gru_layers, hidden_unit=self.gru_hidden_units,
                                     name="attention_gru_decoder")


class TransformerSeq2SeqTranslateModel(TranslateModelBase):

    def __init__(self, enc_vocab_size: int, enc_embedding_size, dec_vocab_size, dec_embedding_size,
                 dropout=0.1, num_blocks: int = 2, hidden_units: int = 32, ffn_num_hidden: int = 64,
                 num_head: int = 4,
                 name="translate_model", **kwargs):
        self.enc_vocab_size: int = enc_vocab_size
        self.dec_vocab_size: int = dec_vocab_size
        self.num_blocks: int = num_blocks
        self.dropout = dropout
        self.hidden_units: int = hidden_units
        self.ffn_num_hidden: int = ffn_num_hidden
        self.num_head: int = num_head
        self.dec_embedding_size: int = dec_embedding_size
        self.enc_embedding_size: int = enc_embedding_size
        super(TransformerSeq2SeqTranslateModel, self).__init__(name=name, **kwargs)

    def build_encoder_layer(self) -> EncoderLayer:
        return TransformerEncoderLayer(enc_vocab_size=self.enc_vocab_size,
                                       embedding_size=self.enc_embedding_size,
                                       num_blocks=self.num_blocks,
                                       dropout=self.dropout,
                                       hidden_unit=self.hidden_units,
                                       ffn_num_hidden=self.ffn_num_hidden,
                                       num_head=self.num_head)

    def build_decoder_layer(self) -> DecoderLayer:
        return TransformerDecoderLayer(dec_vocab_size=self.dec_vocab_size,
                                       embedding_size=self.dec_embedding_size, dropout=self.dropout,
                                       num_blocks=self.num_blocks,
                                       hidden_unit=self.hidden_units,
                                       ffn_num_hidden=self.ffn_num_hidden,
                                       num_head=self.num_head)
