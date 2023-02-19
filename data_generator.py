#!/usr/bin/env python3
# -*- coding:utf-8 _*-
"""
@file: data_generator
@author: jkguo
@create: 2023/2/16
"""
import typing

import numpy as np
import tensorflow as tf
from vocab import Vocab, BEGIN_SENTENCE, END_SENTENCE, PAD_SENTENCE


class TranslateDataGenerator(tf.keras.utils.Sequence):

    def __init__(self, x: typing.List[str], y: typing.List[str]
                 , x_sentence_len: int, y_sentence_len: int
                 , x_token_mode="word", y_token_mode="word"
                 , min_freq=5, x_vocab: Vocab = None, y_vocab: Vocab = None):
        """
        :param x: 源语言字符串
        :param y: 翻译后的译文字符串
        :param x_sentence_len: 源字符串的语句长度，超过大于的则截断
        :param y_sentence_len: 翻译后字符串语句长度，超过则截断
        :param min_freq: 最小token频次，小于min_freq则被认为<unk>
        :param x_vocab: 源语言的词典
        :param y_vocab: 翻译语言的词典
        """
        self.reserved_tokens = [
            BEGIN_SENTENCE,
            END_SENTENCE,
            PAD_SENTENCE
        ]
        # 源语言
        self.x: typing.List[str] = x
        # 目标语言
        self.y: typing.List[str] = y
        # 源字符串的语句长度，超过大于的则截断
        self.x_sentence_len: int = x_sentence_len
        # 翻译后字符串语句长度，超过则截断
        self.y_sentence_len: int = y_sentence_len
        # 用于编码器的输入 shape [sample_count, x_sentence_len] type: int
        self.enc_x: np.ndarray = np.array([])
        # enc_x 每个样本的字符串长度 shape [sample_count]
        self.enc_valid_len: shape[sample_count]
        # 用于解码器的输入 shape [sample_count, y_sentence_len] type int
        self.dec_x: np.ndarray = np.array([])
        # 目标label， 用于验证输出 shape [sample_count, y_sentence_len]
        self.target_y: np.ndarray = np.array([])
        # dec_x 每个样本的字符串长度 shape [sample_count]
        self.dec_valid_len: np.ndarray = np.array([])
        # 输入语言的词典
        self.x_vocab = x_vocab
        if self.x_vocab is None:
            print("building source vocab...")
            self.x_vocab = Vocab(x, token_mode=x_token_mode, min_freq=min_freq, reserved_tokens=self.reserved_tokens)
        # 翻译后语言的词典
        self.y_vocab = y_vocab
        if self.y_vocab is None:
            print("building targe vocab...")
            self.y_vocab = Vocab(y, token_mode=y_token_mode, min_freq=min_freq, reserved_tokens=self.reserved_tokens)
        self.__convert_format()

    def summary(self):
        print(f"case count: {len(self.x)}")
        print("source: ")
        print(f"  vocab size: {len(self.x_vocab)}")
        print(f"  enc_x avg len: {np.average(self.enc_valid_len)}")
        print("targe: ")
        print(f"  vocab size: {len(self.y_vocab)}")
        print(f"  dec_x avg len: {np.average(self.dec_valid_len)}")

    @staticmethod
    def __str_to_vec(lines: typing.List[str], max_len: int, vocab: Vocab):
        valid_len = []
        table = []
        for line in lines:
            if vocab.token_mode == "word":
                line = line.strip().split()
            if len(line) > max_len:
                line = line[: max_len]
            r = [
                vocab[token] for token in line
            ]
            valid_len.append(len(r))
            if len(line) < max_len:
                cnt = max_len - len(line)
                r.extend([vocab[PAD_SENTENCE]] * cnt)
            table.append(r)
        return np.array(table, dtype=np.int32), np.array(valid_len)

    def __convert_format(self):
        """
        转换为模型的输入，输出
        :return:
        """
        print("converting x to enc_x...")
        self.enc_x, self.enc_valid_len = self.__str_to_vec(
            self.x, self.x_sentence_len, self.x_vocab
        )
        print("converting y to dec_x...")
        self.target_y, self.dec_valid_len = self.__str_to_vec(
            self.y, self.y_sentence_len, self.y_vocab
        )
        sample_count = self.target_y.shape[0]
        bos = np.array([self.y_vocab[BEGIN_SENTENCE]] * sample_count).reshape(-1, 1)
        self.dec_x = np.concatenate([bos, self.target_y[:, : -1]], axis=1)
        for i, e in enumerate(self.dec_valid_len):
            if e < self.y_sentence_len:
                self.target_y[i][e] = self.y_vocab[END_SENTENCE]
        self.dec_valid_len += 1

    def __getitem__(self, indices):
        return self.enc_x[indices], self.enc_valid_len[indices], self.dec_x[indices], self.dec_valid_len[indices], \
               self.target_y[indices]

    def __len__(self):
        return len(self.enc_x)
