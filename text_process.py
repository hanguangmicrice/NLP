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
user_dict_path = r'C:\Users\william.han\Downloads\NLP\data\user_dict.txt'

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
                r'[\s+\-\|\!\/\[\]\{\}_,.$%^*(+\"\')]+|[:：+——()?【】“”！，。？、~@#￥%……&*（）]+|车主说|技师说|语音|图片|你好|您好+|(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]',
                   '',sentence)
    else:
        return ''

test_sentence = r'一档起步以后，在未挂二挡的情况下踩离合呜呜呜呜响，随着车速变化而变化，因为一档起步也就一脚油，踩着离合挂到空挡声音消失，进去二挡也没有声音，345也没有，我想问一下这是什么情况。分离轴承吗，假设分离轴承异响，为什么其他档位踩着离合没声音，只有1进2的时候有，谢谢师傅！明天去4s，给处处主意，1.8万公里1年10个月的车，而且有时候一进二咔咔打齿，基本每天都会遇到,技师说：您好，根据您的描述 踩住离合器出现的异响就是分离轴承 ，低速时候转的慢 比较明显 ，高速时候不是没有声音 而是由于发动机转速提高 声音变得连续 不明显了  挂档有打齿的声音除了离合器没有踩到底就是变速箱里面一二档同步器齿环有问题  由于是新车建议ssss免费质保 索赔 进行维修|车主说：二挡我没提高转速，也没有异响|车主说：一档踩着离合，挂到空挡就不响|技师说：抬起来离合有异响吗|技师说：只是踩住时候异响吗|车主说：是的，抬起来或者挂到空挡都没声音|技师说：嗯，这样的话还是分离轴承 ，只有踩住离合器时候分离轴承才会工作 ，不管别的档位 只要踩住离合器有异响 放开没有 就是分离轴承|车主说：而且1换2的时候顿挫感比较强，总是进2档坷垃的声音|技师说：确定离合器踩到底的话 就是变速箱里面一二档同步器齿环的问题 ，需要拆检变速箱|技师说：更换|车主说：我就纳闷的是我二挡也没提高转速，都是回落到一档的转速进二挡，踩着离合也没声音，为啥一档踩着离合就有声音，，呜呜呜呜呜呜呜呜跟车速变化，直到车停了|车主说：我车踩一万八，一万二的时候换过盘和片|技师说：离合片 压盘 表面不平 分离轴承低速运转时候接触不平 也会造成这个现象|技师说：尽快ssss索赔 质保|技师说：重新更换离合器三件套 包括离合片 分离轴承 压盘 并拆检变速箱 更换一二档同步器齿环|车主说：我压盘和离合器片换了才五千公里|车主说：我二挡打齿也不是次次犯，不好索赔啊|技师说：只能说没有磨合好 或者质量不好|技师说：二挡打齿现象如果不频繁的话只能先使用 ，等故障明显时候一起去索赔|车主说：师傅你给判定一下，我一档起步，踩离合到底，档位1没动，此时变速箱内部是什么情况|车主说：就是准备升档，只踩了离合器|车主说：在不在师傅|技师说：在|技师说：这个时候变速箱里面有一半齿轮在旋转|车主说：会不会是变速箱里出现的声音，踩着离合挂空挡声音消失，此时分离轴承一直没运动|车主说：运动的有哪几个齿轮啊|车主说：或者轴承|技师说：这个时候已经切断了变速箱的传动 运转的齿轮也是没有负荷的 相当于空转|技师说：所以说 这个时候就是变速箱里面有异响也是听不出来的|技师说：这就是为什么踩住离合器车辆声音都会变小 的原因|车主说：谢谢了，我去ssss听听他们怎么忽悠我|技师说：不客气|技师说：呵呵|车主说：谢谢|技师说：不客气|车主说：http://v.youku.com/v_show/id_XMTczNTY4MzU2OA==.html 您复制一下，在浏览器打开，我录的视频，你看看什么毛病，一档准备升二的时候踩下离合器，档位还是1|技师说：打不开Q694,上汽通用雪佛兰,迈锐宝XL,倒车打方向盘的时候有响声。,技师说：你好，检查装护板了吗？检查底盘的三角臂有松框的吗？|车主说：发动机护板吗？|车主说：装了|技师说：你好，可以看看护板螺母有松的吗？'

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
jieba.load_userdict(user_dict_path)
words = list(jieba.cut(test_sentence))
print(words)
print(filter_stop_words(words,stop_words))

#combine all the above preprocess 
def sentence_proc(sentence,stop_words=stop_words):
    sentence= clean_sentence(sentence)
    jieba.load_userdict(user_dict_path)
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