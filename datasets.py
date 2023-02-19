#!/usr/bin/env python3
# -*- coding:utf-8 _*-
"""
@file: datasets
@author: jkguo
@create: 2023/2/12
"""
import json
import os
import requests
from zipfile import ZipFile

D2L_DATA_URL = "http://d2l-data.s3-accelerate.amazonaws.com/"


class EnglishFrenchTranslateDatasets(object):

    def __init__(self, data_dir="./data", dataset_name="fra-eng"):
        self.data_zip_file_name: str = f"{dataset_name}.zip"
        self.data_remote_url: str = os.path.join(D2L_DATA_URL, self.data_zip_file_name)
        self.local_data_zip_file_path: str = os.path.join(data_dir, os.path.basename(self.data_remote_url))
        self.local_data_dir: str = os.path.join(data_dir,
                                                os.path.splitext(os.path.basename(self.local_data_zip_file_path))[0]
                                                )
        self.fra_file_path = os.path.join(self.local_data_dir, "fra.txt")

    def load_train_data(self):
        return self.load_data()

    def load_test_data(self):
        return self.load_data()

    def load_data(self):
        self.__prepare()
        x = []
        y = []
        with open(self.fra_file_path, "r", encoding="utf-8") as fp:
            for line in fp.readlines():
                line = self.preprocess_nmt(line.strip())
                parts = line.split("\t")
                if len(parts) == 2:
                    x.append(parts[0])
                    y.append(parts[1])
        return x, y

    @staticmethod
    def preprocess_nmt(text):
        """预处理“英语－法语”数据集"""

        def no_space(char, prev_char):
            return char in set(',.!?') and prev_char != ' '

        # 使用空格替换不间断空格
        # 使用小写字母替换大写字母
        text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
        # 在单词和标点符号之间插入空格
        out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
               for i, char in enumerate(text)]
        return ''.join(out)

    def __prepare(self):
        if os.path.isfile(self.fra_file_path):
            return
        # download from web
        if not os.path.isfile(self.local_data_zip_file_path):
            print(f"downloading dataset {self.data_remote_url}")
            response = requests.get(self.data_remote_url)
            open(self.local_data_zip_file_path, "wb").write(response.content)
            print(f"save to {self.local_data_zip_file_path} success.")
        # extract zip file
        with ZipFile(self.local_data_zip_file_path, 'r') as f:
            f.extractall(os.path.dirname(self.local_data_dir))


class EnglishChineseTranslateDatasets(object):
    ENGLISH_SPECIAL_REPLACE_CHAR_MAP = {
        '“': '"',
        '”': '"',
        '‘': '\'',
        '’': '\'',
        '…': '.',
        '（': '(',
        '）': ')',
        '—': '-',
        '–': '-',
        '·': '*',
        '。': '.',
        '＄': '$',
        '】': ']',
        '【': '['
    }

    def __init__(self, data_dir="./data", dataset_name="translation2019zh", need_format=False):
        self.train_file: str = os.path.join(data_dir, f"{dataset_name}/{dataset_name}_train.json")
        self.test_file: str = os.path.join(data_dir, f"{dataset_name}/{dataset_name}_valid.json")
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None
        self.need_format = need_format

    def load_train_data(self):
        if self.train_x is None or self.train_y is None:
            self.train_x, self.train_y = self.load_data(self.train_file)
        return self.train_x, self.train_y

    def load_test_data(self):
        if self.test_x is None or self.test_y is None:
            self.test_x, self.test_y = self.load_data(self.test_file)
        return self.test_x, self.test_y

    def load_data(self, file_path: str):
        x, y = [], []
        with open(file_path, "r") as fp:
            for line in fp.readlines():
                if len(line) == 0:
                    continue
                doc = json.loads(line)
                en = doc["english"]
                ch = doc["chinese"]
                if self.need_format:
                    if EnglishChineseTranslateDatasets.is_need_swap(en, ch):
                        en, ch = ch, en
                    en = EnglishChineseTranslateDatasets.__format_english(en)
                    if en.startswith('<form action="join.jsp"'):
                        continue
                x.append(en)
                y.append(ch)
        return x, y

    @staticmethod
    def __format_english(en: str):
        need_replace = False
        for idx, ch in enumerate(en):
            if ch in EnglishChineseTranslateDatasets.ENGLISH_SPECIAL_REPLACE_CHAR_MAP:
                need_replace = True
                break
        if need_replace:
            format_en = ""
            for idx, ch in enumerate(en):
                if ch in EnglishChineseTranslateDatasets.ENGLISH_SPECIAL_REPLACE_CHAR_MAP:
                    format_en += EnglishChineseTranslateDatasets.ENGLISH_SPECIAL_REPLACE_CHAR_MAP[ch]
                else:
                    format_en += ch
            en = format_en
        return en

    @classmethod
    def __not_english_char(cls, ch):
        return ch > chr(127) and ch not in ['《', 'π', 'Ê']

    @classmethod
    def is_need_swap(cls, en_s: str, chinese_s: str):
        en_cnt = 0
        mx_e = chr(127)
        for ch in en_s:
            if ch < mx_e:
                en_cnt += 1
        en_r = en_cnt * 1.0 / len(en_s)
        ch_cnt = 0
        for ch in chinese_s:
            if ch < mx_e:
                ch_cnt += 1
        ch_r = ch_cnt * 1.0 / len(chinese_s)
        return en_r < ch_r

    def save(self):
        if self.train_x is not None and len(self.train_x) == len(self.train_y):
            with open(self.train_file, "w") as fp:
                for i in range(len(self.train_x)):
                    doc = {
                        "english": self.train_x[i],
                        "chinese": self.train_y[i]
                    }
                    fp.write(json.dumps(doc))
                    fp.write('\n')
        if self.test_x is not None and len(self.test_x) == len(self.test_y):
            with open(self.test_file, "w") as fp:
                for i in range(len(self.test_x)):
                    doc = {
                        "english": self.test_x[i],
                        "chinese": self.test_y[i]
                    }
                    fp.write(json.dumps(doc))
                    fp.write('\n')
