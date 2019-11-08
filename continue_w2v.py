from gensim.models import word2vec
from gensim.models.word2vec import LineSentence
model=word2vec.Word2Vec.load('细粒度词向量.model')
model.build_vocab(sentences=LineSentence('验证集评论.txt'),update=True)
model.train(sentences=LineSentence('验证集评论.txt'),total_examples=model.corpus_count,epochs=model.iter)
model.save('all_vectors.model')
