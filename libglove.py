from tqdm import tqdm
from nltk.tokenize import word_tokenize
import numpy as np
name='Glove 840B 300d'
print("Loading Glove 840B Word Vectors")
embedding_file = '/home/saradhix/GloVe_pretrained/glove.840B.300d.txt'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in tqdm(open(embedding_file),total=2196017))
#max_features=len(embeddings_index.values()[0])
max_features=len(list(embeddings_index.values())[0])

def get_all_embeddings():
  return embeddings_index
def get_vector(sentence):
    final_vector = np.zeros(max_features)
    total = 0
    #words = sentence.split(' ')
    words = [word for word in word_tokenize(sentence) if word.isalpha()]
    for word in words:
        vector = embeddings_index.get(word.lower())
        if vector is not None:
            total = total+1
            final_vector = final_vector + vector
        if total ==0:
            total=1 #To prevent div by 0
        #final_vector = final_vector / total
    return final_vector

def get_vectors(sentences):
    final_vectors=[]
    for sentence in tqdm(sentences):
        vector = get_vector(sentence)
        final_vectors.append(vector)
    return np.array(final_vectors)

def get_vectors_sel(sentences, filtered_words):
    final_vectors=[]
    for sentence in tqdm(sentences):
        vector = get_vector_sel(sentence, filtered_words)
        final_vectors.append(vector)
    return final_vectors

def get_vector_sel(sentence, filtered_words):
    final_vector = np.zeros(max_features)
    total = 0
    #words = sentence.split(' ')
    words = [word for word in word_tokenize(sentence) if word.isalpha()]
    for word in words:
        if word not in filtered_words: continue
        vector = embeddings_index.get(word.lower())
        if vector is not None:
            total = total+1
            final_vector = final_vector + vector
        if total ==0:
            total=1 #To prevent div by 0
        final_vector = final_vector / total
    return final_vector


