# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 14:13:11 2020

@author: William.Han
"""

import pandas as pd
import re
import jieba
import numpy as np
from multiprocessing import cpu_count, Pool

Train_file_path = r'C:\Users\william.han\Downloads\NLP\data\AutoMaster_TrainSet.csv'
Test_file_path = r'C:\Users\william.han\Downloads\NLP\data\AutoMaster_TestSet.csv'
Stop_word_path=  r'C:\Users\william.han\Downloads\NLP\data\stopwords\哈工大停用词表.txt'

##load train and test data

def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path,encoding='utf-8')
    test_df = pd.read_csv(test_path,encoding='utf-8')
    return train_df, test_df

train_data, test_data = load_data(Train_file_path, Test_file_path)
train_data.info()

##drop the nan value
train_data.dropna(subset=['Question', 'Dialogue', 'Report'], how='any', inplace=True)
test_data.dropna(subset=['Question','Dialogue'],how='any', inplace=True)

##clean useless chacter in sentence
def clean_sentence(sentence):
    if isinstance(sentence,str):
        return re.sub(
                r'[\s+\-\|\!\/\[\]\{\}_,.$%^*(+\"\')]+|[:：+——()?【】“”！，。？、~@#￥%……&*（）]+|车主说|技师说|语音|图片|你好|您好',
                   '',sentence)
    else:
        return ''

test_sentence = r'技师说：[语音]|车主说：新的都换了|车主说：助力泵，方向机|技师说：[语音]|车主说：换了方向机带的有|车主说：[图片]|技师说：[语音]|车主说：有助力就是重，这车要匹配吧|技师说：不需要|技师说：你这是更换的部件有问题|车主说：跑快了还好点，就倒车重的很。|技师说：是非常重吗|车主说：是的，累人|技师说：[语音]|车主说：我觉得也是，可是车主是以前没这么重，选吧助理泵换了不行，又把放向机换了，现在还这样就不知道咋和车主解释。|技师说：[语音]|技师说：[语音]'

print(clean_sentence(test_sentence))

##cut the sentence
def load_stop_dict(file_path):
    f = open(file_path,'r' ,encoding='utf-8')
    lines = f.readlines()
    stop_words_list = [s.strip() for s in lines]
    return stop_words_list

stop_words = load_stop_dict(Stop_word_path)

def filter_stop_words(sentence,stop_words):
    return [i for i in sentence if i not in stop_words]

test_sentence='2010款的宝马X1，2011年出厂，2.0排量'
words = list(jieba.cut(test_sentence))
print(words)
print(filter_stop_words(words,stop_words))

#combine all the above preprocess 
def sentence_proc(sentence,stop_words=stop_words):
    sentence= clean_sentence(sentence)
    words = list(jieba.cut(sentence))
    words = filter_stop_words(words,stop_words)
    return ' '.join(words)

print(sentence_proc(test_sentence,stop_words))

#process the df one time
def df_proc(df, stop_words):
    for i in  ['Brand', 'Model', 'Question', 'Dialogue']:
        print(i)
        df[i] = df[i].apply(sentence_proc)
    if 'Report' in df.columns:
        df['Report'] = df['Report'].apply(sentence_proc)
    return df

train_data= df_proc(train_data,stop_words)
test_data= df_proc(test_data,stop_words)
#use pallel cpu to speed up
# cpu no
cores = cpu_count()
# partition no.
partitions = cores 

def parallelize(df, fn,cores,partitions):
    # data split
    data_split = np.array_split(df,partitions)
    # pool
    pool = Pool(cores)
    # data split and concat
    df = pd.concat(pool.map(fn,data_split))
    # close pool
    pool.close()
    pool.join()
    return df

train_data = parallelize(train_data,df_proc,cores,partitions)
## save the data
train_data.to_csv(r'C:\Users\william.han\Downloads\NLP\data\train_seg_data.csv', index=None, header=True)
test_data.to_csv(r'C:\Users\william.han\Downloads\NLP\data\test_seg_data.csv', index=None, header=True)


#build the vocab
train_data['merged'] = train_data[['Question', 'Dialogue','Report']].apply(lambda x:' '.join(x), axis=1)
test_data['merged'] = test_data[['Question', 'Dialogue']].apply(lambda x:' '.join(x), axis=1)

df_merge = pd.concat([train_data[['merged']],test_data[['merged']]], axis=0)
df_merge.to_csv(r'C:\Users\william.han\Downloads\NLP\data\merged_train_test_seg_data.csv',index=None,header=False)

voca = set(' '.join(df_merge['merged']).split())