#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
功能：fasttext学习
时间：2018年05月03日09:17:28
函数：cbow，load_model，skipgram，supervised
版本：0.8.3
"""

import fasttext
import json

# Skipgram model
# model = fasttext.skipgram('/Users/simon/Downloads/news_fasttext_train.txt', 'model')
# print(model.words)  # list of words in dictionary

# CBOW model
# model = fasttext.cbow('/Users/simon/Downloads/news_fasttext_train.txt', 'model')
# print(model.words)  # 词表

model = fasttext.load_model('model.bin')  # 加载模型
print(model['北京'])  # 获取词向量


def save_words_01():
    """保存词表，方式一"""
    with open("词表.txt", "w+") as fout:
        for word in model.words:
            fout.write(word + "\n")


def save_words_02():
    """保存词表，方式二"""
    fout = open("results/词表2.json", "w+")
    words = []
    with open("model.vec") as f:
        for line in f:
            temp = line.rstrip().split(" ")
            if len(temp) > 2:
                words.append(temp[0])

    json.dump(words, fout, ensure_ascii=False, indent=4)
    fout.close()


if __name__ == "__main__":
    save_words_02()
