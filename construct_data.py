import torch
import numpy as np
from gensim.models import word2vec
from torch.utils.data.dataset import Dataset
import torch.utils.data as d
model=word2vec.Word2Vec.load('word2vec训练好的词向量.model')#此处的加载的model文件需对应自己的任务进行训练
class trainData(Dataset):
    def __init__(self,file_path,label_path):
        self.file_path=file_path
        self.label_path=label_path
        file1=open(self.file_path,'r')
        file2=open(self.label_path,'r')
        self.comment=[]   #此处就是将训练数据示例.txt中的4条文本放在列表中，如果您有10000条，那么将10000条文本放入到列表中
        for line in file1.readlines():
            self.comment.append(line)
        self.labels=[]    #此处是将对应的标签放入到列表中
        for num in file2.readlines():
            self.labels.append(num)
    def __getitem__(self,index):
        data=self.comment[index]  #此处的index即会循环遍历self.comment中的所有文本
        label=self.labels[index]  #此处是对应的标签
        data=data.split(' ')      #以下内容根据自身任务自行书写
        com=np.zeros((300,200))
        if len(data)<=300:
            i=0
            for word in data:
                if word !='\n':
                    com[i]=model[word]
                    i+=1
        else:
            i=0
            for word in data:
                if word != '\n':
                    com[i]=model[word]
                    i+=1
                    if i ==300:
                        break
        #print(com)
        lab=[]
        label=label.split()
        j=0
        for num in label:
            if num == '1':
                lab.append(0)
            elif num == '0':
                lab.append(1)
            elif num == '-1':
                lab.append(2)
            else:
                lab.append(3)
            j+=1
        return torch.FloatTensor(com),torch.Tensor(lab)    #此处返回的是构建好一条数据及其对应的标签。
    def __len__(self):    #此方法返回共有多少个数据
        return len(self.labels)
train_data=trainData('训练数据示例.txt','训练数据标签.txt')
train_loader=d.DataLoader(dataset=train_data,batch_size=2,shuffle=True)#通过dataloader方法可以批量导入数据，将数据输入到神经网络。
for step,(x,y) in enumerate(train_loader):
    print(x,x.shape)
    print(y,y.shape)
