# -*- coding: UTF-8 -*-
import pandas as pd
import jieba
import re
jieba.load_userdict('All_hownet.txt')
df=pd.read_csv('/home/gzc/情感细粒度评论数据/训练集/sentiment_analysis_trainingset.csv',encoding='utf-8')
stopkey=[]
stop_file=open('stopword.txt','r',encoding='gb18030')
stop_words=stop_file.readlines()
for stop_word in stop_words:
    stopkey.append(stop_word.strip())
#print(stopkey)
#print(stop_file.readlines())
f=open('细粒度评论.txt','a+')
sentences=df['content']
for sen in sentences:
    sens=re.compile(u"[\u4e00-\u9fa5]")#正则化处理，只提出汉字
    res=re.findall(sens,sen)
    ss=''.join(res)
    ss=jieba.lcut(ss,cut_all=False)
    #print(ss)
    for word in ss:
        if word not in stopkey and word!='\n' and word!='\r':
            f.write(word+' ')
    f.write('\n')
f.close()
stop_file.close()
             
