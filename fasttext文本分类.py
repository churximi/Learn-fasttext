#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
功能：fasttext学习
时间：2018年05月02日15:45:05
"""

import logging
import fasttext

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# 训练模型
train_file = "/Users/simon/Downloads/news_fasttext_train.txt"
test_file = "/Users/simon/Downloads/news_fasttext_test.txt"
save_model_file = "models/model"
# classifier = fasttext.supervised(train_file, save_model_file, label_prefix="__label__")

# load训练好的模型
classifier = fasttext.load_model('models/news_fasttext.model.bin', label_prefix='__label__')

# 测试模型
result = classifier.test(test_file)
print("准确率", result.precision)
print("召回率", result.recall)
print("实例个数：", result.nexamples)

# 预测新数据标签
new_data = ['这是 一篇 体育 新闻 ， 足球 比赛 ...', '政治 外交 美国', '政治 外交 中国']
labels = classifier.predict(new_data)
print(labels)

# 或者同时输出概率
labels = classifier.predict_proba(new_data)
print(labels)

# 也可以指定输出最可能的标签数
labels = classifier.predict(new_data, k=3)
print(labels)

# Or with the probability
labels = classifier.predict_proba(new_data, k=3)
print(labels)

if __name__ == "__main__":
    pass
