import torch
from torch import nn
import torch.utils.data as d
from torch.utils.data.dataset import Dataset
import numpy as np
from tensorboardX import SummaryWriter as S
import torch.nn.functional as F
from construct_data import traingzc
#from test_data import testData
from Single_channel_RCNN import SCRCNN
from predata import DIC
from gensim.models import word2vec
from Attention import Attn
model=word2vec.Word2Vec.load('all_vectors.model')
word_name=['负面评价词语（中文）.txt','负面情感词语（中文）.txt','正面评价词语（中文）.txt','正面情感词语（中文）.txt']
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
train_data=traingzc(w2i,i2w,'细粒度评论.txt','细粒度标签.txt',A,B,C,D)
train_loader=d.DataLoader(dataset=train_data,batch_size=5,shuffle=True)
'''test_data=testData('验证集评论.txt','验证集标签.txt')
test_loader=d.DataLoader(dataset=test_data,batch_size=500,shuffle=True)'''
EPOCH = 1
TIME_STEP = 300
INPUT_SIZE = 200
word_embedding_size=200
Dictionary_size=266440
word_feature_embedding=50
LR=0.01
w2v=np.zeros((Dictionary_size,word_embedding_size))
j=1
for i in i2w:
    if i==0:
        continue
    else:
        w2v[j]=model[i2w[i]]
        j+=1 
w2v=torch.from_numpy(w2v)
writer=S(log_dir='./模型')
class MCRCNN(nn.Module):
    def __init__(self,w2v_weight):
        super(MCRCNN,self).__init__()
        self.w2v_weight=w2v_weight
        #print(self.w2v_weight.shape)
        self.channel1=SCRCNN(self.w2v_weight,word_embedding_size)
        self.channel2=SCRCNN(None,word_embedding_size+word_feature_embedding)
        self.attn=Attn(128)    #此处的128是根据Single_channel_RCNN中最后一层全连接层大小而来
        self.L1=nn.Linear(128,64)  #此处为认为设置多通道合并之后的全新链接层大小，根据上一层获得
        self.dropout=nn.Dropout(p=0.4)
        self.L2=nn.Linear(64,4)
    def forward(self,b_x,b_y):
        x1,y=self.channel1(b_x,b_y,add_feature=False)   #x1.shape(batch_size*128)
        x2  =self.channel2(b_x,b_y,add_feature=True)[0] #x2.shape(batch_size*128)
        '''print('x1:',x1,x1.shape)
        print('--------------')
        print('x2:',x2,x2.shape)'''
        X1=x1.view(-1,1,128)
        X2=x2.view(-1,1,128)
        X=torch.cat((X1,X2),1)
        #print('X:',X.shape)   #X为x1一行紧接着x2一行这种排列方式，因为x1与x2代表不同的通道提取的特征。
        score=self.attn(X)
        context=torch.zeros((X.shape[0],128))
        for i in range(X.shape[0]):
            context[i]=x1[i]*score[i][0]+x2[i]*score[i][1]
        #print('context:',context.shape)   context是将每个通道提取特征之后得到的注意力机制的结果。
        context=self.L1(context)
        context=self.dropout(context)
        X=self.L2(context)
        return X,y
mcrcnn=MCRCNN(w2v)
print(mcrcnn)
optimizer=torch.optim.SGD(mcrcnn.parameters(),lr=LR)
loss_func=nn.CrossEntropyLoss()
for epoch in range(EPOCH):
    for step,(b_x,b_y) in enumerate(train_loader):
        #print(b_x,b_x.shape)
        output,labels=mcrcnn(b_x,b_y)
        #print(labels.shape)
        loss=loss_func(output,labels)
        print(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
