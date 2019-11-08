import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
class Attn(nn.Module):
    def __init__(self,hidden_size):
        super(Attn,self).__init__()
        self.hidden_size=hidden_size
        self.L=nn.Linear(self.hidden_size,1)
    def forward(self,hidden_state):
        score=self.L(hidden_state)
        score=score.squeeze(2)
        #print(score,score.shape)
        return F.softmax(score,dim=1)
'''a=torch.randn(2,3,4)
att=Attn(4)
s=att(a)
print(s)'''
