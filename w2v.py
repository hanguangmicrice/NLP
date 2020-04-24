# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 16:23:30 2020

@author: William.Han
"""

import gensim
import pandas as pd
from gensim.models.word2vec import LineSentence
from gensim.models import word2vec
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
seg_word_path = r'C:\Users\william.han\Downloads\NLP\data\merged_train_test_seg_data.csv'

merged_df = pd.read_csv(seg_word_path,header=None)

"""
Gensim中 Word2Vec 模型的期望输入是分词后的句子列表，即是某个二维数组。这里我们暂时使用 Python 内置的数组，不过其在输入数据集较大的情况下会占用大量的 RAM。Gensim 本身只是要求能够迭代的有序句子列表，因此在工程实践中我们可以使用自定义的生成器，只在内存中保存单条语句。

size:神经网络的层数 默认100

workers:线程数量

min_count: 去掉低频词

"""

model = word2vec.Word2Vec(LineSentence(seg_word_path), size=200, workers=8,min_count=5)

model.wv.most_similar(['奔驰'], topn=10)
save_modelk_path=r'C:\Users\william.han\Downloads\NLP\data\word2vec.model'
model.save(save_modelk_path)
