from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
model=Word2Vec(LineSentence('细粒度评论.txt'), 
                  size=200, 
                  window=5,
                  min_count=1, 
                  workers=2,
                  negative=5,
                  iter=10)
model.save('细粒度词向量.model')
model=Word2Vec.load('细粒度词向量.model')
#print(model['第一次'])

