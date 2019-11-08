from predata import DIC
import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import torch.utils.data as d
class traingzc(Dataset):
    def __init__(self,word2index,index2word,file_path,label_path,NW,NS,PW,PS):
        self.word2index=word2index
        self.index2word=index2word
        self.file_path=file_path
        self.label_path=label_path
        self.NW=NW   #负面词语
        self.NS=NS   #负面情感
        self.PW=PW   #正面词语
        self.PS=PS   #正面情感
        self.Max_length=300
        file1=open(self.file_path,'r')
        file2=open(self.label_path,'r')
        self.comment=[]
        for line in file1.readlines():
            self.comment.append(line)
        self.labels=[]
        for num in file2.readlines():
            self.labels.append(num.split(' ')[0])
    def __getitem__(self,index):
        data=self.comment[index]
        label=self.labels[index]
        sen=data.split(' ')
        sens=[]
        word_features=[]
        for word in sen:
            if word!='\n':
                sens.append(self.word2index[word])
                if word in self.NW:
                    word_features.append(1)
                elif word in self.NS:
                    word_features.append(2)
                elif word in self.PW:
                    word_features.append(3)
                elif word in self.PS:
                    word_features.append(4)
                else:
                    word_features.append(5)
        if len(sens)<self.Max_length:
            for i in range(self.Max_length-len(sens)):
                sens.append(0)
                word_features.append(0)
        else:
            sens=sens[:self.Max_length]
            word_features=word_features[:self.Max_length]
        if label == '1':
            label=0
        elif label == '0':
            label=1
        elif label == '-1':
            label=2
        else:
            label=3
        return torch.LongTensor([sens,word_features]),torch.Tensor([label]).squeeze()
    def __len__(self):
        return len(self.labels)
'''word_name=['负面评价词语（中文）.txt','负面情感词语（中文）.txt','正面评价词语（中文）.txt','正面情感词语（中文）.txt']
word_list=[[],[],[],[]]
i=0
for x in word_list:
    f=open('/home/gzc/pytorh练习/multi-channel-Rcnn/hownet/'+word_name[i],'r')
    for line in f.readlines():
        x.append(line.strip('\n'+' '))
    i+=1
A=word_list[0]
B=word_list[1]
C=word_list[2]
D=word_list[3]
dic=DIC('测试评论+验证评论.txt')
w2i,i2w=dic.addsentence()
print(len(i2w))'''
'''train_data=traingzc(w2i,i2w,'细粒度评论.txt','细粒度标签.txt',A,B,C,D)
train_loader=d.DataLoader(dataset=train_data,batch_size=5,shuffle=True)
for step,(x,y) in enumerate(train_loader):
    print(x,x.shape)
    print(y,len(y))'''





















