import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from Attention import Attn
LSTM_hidden_size=128     #LSTM隐藏层大小
Dictionary_size=266440   #字典中第一位为0，意义为padding
word_embedding_size=200
Sequence_max_length=300
word_feature_embedding=50
class SCRCNN(nn.Module):
    def __init__(self,weight,input_size):
        super(SCRCNN,self).__init__()
        self.weight=weight
        self.input_size=input_size
        self.embedding1=nn.Embedding(Dictionary_size,word_embedding_size,padding_idx=0,_weight=self.weight)
        self.embedding2=nn.Embedding(6,word_feature_embedding,padding_idx=0)  #此处的6为只有5种词性，还有0作为padding，50为词性特征向量化大小
        self.lstm=nn.LSTM(input_size=self.input_size,
                          hidden_size=LSTM_hidden_size,
                          num_layers=2,
                          batch_first=True,
                          dropout=0,
                          bidirectional=True)
        self.linear=nn.Linear(self.input_size+2*LSTM_hidden_size,128)
        self.tanh=nn.Tanh()
    def rep_text(self,a,b,c,d):
        a=a.view(len(b),b[0],2,LSTM_hidden_size)
        mat_FW=torch.zeros((a.shape[0],b[0],LSTM_hidden_size))
        for i in range(a.shape[0]):
            mat_FW[i][0]=d
            for l in range(1,b[i]):
                mat_FW[i][l]=a[i][l][0]
        mat_BW=torch.zeros((a.shape[0],b[0],LSTM_hidden_size))
        for j in range(a.shape[0]):
            mat_BW[j][b[j]-1]=d
            for k in range(b[j]-1):
                mat_BW[j][k]=a[j][b[j]-2-k][1]
        data=torch.cat((mat_FW,c),2)
        data=torch.cat((data,mat_BW),2)
        return data                #rep_text方法是为了获得RCNN中排序好的矩阵
    def packedd(self,x,y):
        x=x.view(-1,Sequence_max_length,self.input_size)
        #print(x.shape)
        x=x.detach().numpy()
        if torch.is_tensor(y) == True:
            y=y.numpy()
        num=[]
        for i in range(x.shape[0]):
            k=0
            for j in range(Sequence_max_length):
                if x[i][j].any()==True:
                    k+=1
            num.append(k)
        lengths=sorted(num,reverse=True)
        lengths_2=lengths
        matrix=np.zeros((x.shape[0],lengths[0],self.input_size))
        label=[]
        for i in range(x.shape[0]):
            matrix[i][:max(num)]=x[num.index(max(num))][:max(num)]
            label.append(y[num.index(max(num))])
            elment=num.index(max(num))
            num[elment]=0
        label=torch.LongTensor(label)
        matrix=torch.FloatTensor(matrix)
        lengths=torch.LongTensor(lengths)
        x_packed=nn.utils.rnn.pack_padded_sequence(matrix, lengths=lengths, batch_first=True)
        return x_packed,label,lengths,matrix      #packedd方法是为了处理可变长度
    def forward(self,b_x,b_y,add_feature):
        #print('b_x:',b_x,b_x.shape)
        B_X=torch.chunk(b_x,2,1)  #b_x由两部分组成，第一部分是每个字在字典中的序号，第二部分是每个字在词性字典中序号，应用chunk函数提取出每一个样本的同一行,将字典序号放在一起，将词性序号放在一起。
        #print(B_X)
        b_x=self.embedding1(B_X[0].squeeze())
        '''print(b_x,b_x.shape)
        print('---------------')'''
        if add_feature==True:
            b_x1=self.embedding1(B_X[0].squeeze())
            b_x2=self.embedding2(B_X[1].squeeze())
            b_x=torch.cat((b_x1,b_x2),2)
            #print(b_x.shape)
        x_packed,label,length,mat=self.packedd(b_x,b_y)
        b_x=b_x.view(-1,300,self.input_size)
        np.random.seed(1)
        H_0=np.random.rand(LSTM_hidden_size)
        h_f=torch.FloatTensor(H_0)
        H_0=np.tile(H_0,(4,b_x.shape[0],1))
        H_0=torch.FloatTensor(H_0)
        C_0=H_0
        hidd=(H_0,C_0)
        r_out,(h_n,h_c)=self.lstm(x_packed,hidd)
        '''print(r_out)
        print('-----------------------------')'''
        out = torch.nn.utils.rnn.pad_packed_sequence(r_out, batch_first=True)
        '''print(out[0].shape)
        print('-----------------------------')'''
        out=out[0]
        data=self.rep_text(out,length,mat,h_f)
        data=self.linear(data)
        data=self.tanh(data)
        #print('data:',data.shape)
        data=data.permute(0,2,1)
        #print('data:',data.shape)
        Max1d=nn.MaxPool1d(int(length.detach().numpy()[0]))
        data=Max1d(data).squeeze(2)
        #print('data:',data.shape)
        return data,label
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
