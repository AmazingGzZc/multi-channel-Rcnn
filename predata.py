PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token
UNK_token = 3  # Unkonw of word
class DIC:
    def __init__(self,path):
        self.path=path
        self.word2index={'PAD':'0'}
        self.index2word={0:'PAD'}
        self.num_words=1
    def addword(self,sentence):
        for word in sentence.split(' '):
            if word not in self.word2index and word!='\n':
                self.word2index[word]=self.num_words
                self.index2word[self.num_words]=word
                self.num_words+=1
    def addsentence(self):
        path=self.path
        f=open(path,'r')
        for line in f.readlines():
            self.addword(line)
        return self.word2index,self.index2word
'''dic=DIC('测试评论+验证评论.txt')
A,B=dic.addsentence()
print(len(B))'''

