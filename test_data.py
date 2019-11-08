import torch
import numpy as np
from gensim.models import word2vec
from torch.utils.data.dataset import Dataset
import torch.utils.data as d
model=word2vec.Word2Vec.load('all_vectors.model')
class testData(Dataset):
    def __init__(self,word2index,index2word,file_path,label_path,NW,NS,PW,PS):
        self.word2index=word2index
        self.index2word=index2word
        self.file_path=file_path
        self.label_path=label_path
        self.NW=NW
        self.NS=NS
        self.PW=PW
        self.PS=PS
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
                sens.append(self.word2indes[word])
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
'''test_data=testData('验证集评论.txt','验证集标签.txt')
test_loader=d.DataLoader(dataset=test_data,batch_size=5,shuffle=True)
for step,(x,y) in enumerate(test_loader):
    print(x,x.shape)
    print(y,y.shape)'''
