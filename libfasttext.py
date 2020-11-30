from tqdm import tqdm
from nltk.tokenize import word_tokenize
import numpy as np
import codecs
name = 'FastText 300d'
print("Loading Fast Text Word Vectors")
embedding_file = '/home/saradhix/FastText_pretrained/cc.en.300_clean.vec'
#embedding_file = '/home/saradhix/fasttext/cc.en.300.vec'
embeddings_index = {}
f = codecs.open(embedding_file, encoding='utf-8')
for line in tqdm(f, total=2519371):
    values = line.rstrip().rsplit(' ')
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('found %s word vectors' % len(embeddings_index))

#max_features=len(embeddings_index.values()[0])
max_features=len(values)-1
print("Number of dimensions=", max_features)

def get_all_embeddings():
  return embeddings_index
def get_vector(sentence):
    final_vector = np.zeros(max_features)
    total = 0
    words = [word for word in word_tokenize(sentence) if word.isalpha()]
    for word in words:
        vector = embeddings_index.get(word.lower())
        if vector is not None:
            #print(word.lower(), vector.shape, vector)
            #print(final_vector.shape)
            try:
              final_vector = final_vector + vector
              total = total+1
            except:
              continue
        if total ==0:
            total=1 #To prevent div by 0
        final_vector = final_vector / total
    return final_vector

def get_vectors(sentences):
    final_vectors=[]
    for sentence in tqdm(sentences):
        vector = get_vector(sentence)
        final_vectors.append(vector)
    return np.array(final_vectors)
