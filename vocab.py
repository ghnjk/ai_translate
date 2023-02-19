#!/usr/bin/env python3
# -*- coding:utf-8 _*-
"""
@file: vocab
@author: jkguo
@create: 2023/2/12
"""
import typing
import collections

UNKNOWN_TOKEN = "<unk>"
BEGIN_SENTENCE = "<bos>"
END_SENTENCE = "<eos>"
PAD_SENTENCE = "<pad>"


class Vocab(object):

    def __init__(self, lines: typing.List[str], token_mode="char", min_freq=0, reserved_tokens=None):
        self.token_to_idx: typing.Dict[str, int] = {}
        self.idx_to_tokens: typing.List[str] = []
        self.token_freq: typing.Dict[str, int] = {}
        self.token_mode: str = token_mode
        self.min_freq: int = min_freq
        self.__create_new_token(UNKNOWN_TOKEN)
        if reserved_tokens is not None:
            for token in reserved_tokens:
                self.__create_new_token(token)
        self.__add_tokens_to_dict(self.__tokenize(lines))
        self.unk = self.token_to_idx[UNKNOWN_TOKEN]

    def __len__(self):
        return len(self.idx_to_tokens)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_tokens[indices]
        return [self.to_tokens(idx) for idx in indices]

    def __create_new_token(self, token: str, freq: int = 1):
        """
        在词典中增加一个token 设置对应的标识和频率
        :param token: 词
        :param freq: 词频
        :return: 词对应的下标
        """
        if token in self.token_to_idx:
            self.token_freq[token] += freq
            return self.token_to_idx[token]
        else:
            idx = len(self.idx_to_tokens)
            self.idx_to_tokens.append(token)
            self.token_to_idx[token] = idx
            self.token_freq[token] = freq
            return idx

    def __add_tokens_to_dict(self, tokens: typing.List[str]):
        """
        将所有的tokens添加到词典中
        :param tokens: token list
        :return:
        """
        counts = collections.Counter(tokens)
        counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        for token, freq in counts:
            if freq >= self.min_freq:
                self.__create_new_token(token, freq)

    def __tokenize(self, lines: typing.List[str]) -> typing.List[str]:
        """
        将所有的文本token化
        :param lines: 文本行列表
        :return: 词化后的序列
        """
        if lines is None:
            lines = []
        if self.token_mode == "char":
            tokens = [
                token
                for line in lines
                for token in line
            ]
        elif self.token_mode == "word":
            words = [
                line.split()
                for line in lines
            ]
            tokens = [
                token
                for line in words
                for token in line
            ]
        else:
            raise Exception(f"not support token_mode {self.token_mode}")
        return tokens
