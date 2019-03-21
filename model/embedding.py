from gensim.models import KeyedVectors
import numpy as np

from config import Config

config = Config()

w2v = None


def get_embedding(word):
    word = str(word).lower().strip()
    vector = np.array([0]*config.embedding_size)
    try:
        global w2v
        if not w2v:
            print('loading w2v')
            w2v = KeyedVectors.load_word2vec_format(
                config.embedding_path,
                binary=True
            )

        if word in w2v:
            vector = w2v[word]
    except Exception as e:
        print(e)
    return vector
