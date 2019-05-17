import re

import pandas as pd
import numpy as np
from gensim.models import Word2Vec, Doc2Vec, FastText
from nltk import SnowballStemmer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

from shift_detector.utils.Miscellaneous import print_progress_bar

DIMENSIONS = 50
# DEPRECATED

class Vector:
    def __init__(self, elementList):
        self.vec = [float(i) for i in elementList]

    def __add__(self, other):
        assert (len(self.vec) == len(other.vec))
        out = []
        for i in range(len(self.vec)):
            out.append(self.vec[i] + other.vec[i])
        return Vector(out)

    def __sub__(self, other):
        assert (len(self.vec) == len(other.vec))
        out = []
        for i in range(len(self.vec)):
            out.append(self.vec[i] - other.vec[i])
        return Vector(out)

    def __mul__(self, other):
        assert (len(self.vec) == len(other.vec))
        out = 0
        for i in range(len(self.vec)):
            out += self.vec[i] * other.vec[i]
        return out

    def magnitude(self):
        out = 0
        for i in self.vec:
            out += i ** 2
        return np.math.sqrt(out)

    def multiply(self, other):
        out = []
        for i in self.vec:
            out.append(i * other)
        return Vector(out)

    def dist(self, other):
        assert (len(self.vec) == len(other.vec))
        a = 0
        for i in range(len(self.vec)):
            a += (self.vec[i] - other.vec[i]) ** 2
        return np.math.sqrt(a)

    def cosine_similarity(self, other):
        return (self * other) / (self.magnitude() * other.magnitude())


def make_usable_list(data):
    usable = []
    for i, entry in enumerate(data):
        wordlist = []
        words = re.sub(r'[^\w+\s]|\b[a-zA-Z]\b',' ',entry).split()
        for word in words:
            if word.lower() not in ENGLISH_STOP_WORDS or '':
                wordlist.append(SnowballStemmer('english').stem(word.lower()))
        if wordlist:
            usable.append(wordlist)
    return usable


def separate_words(data):
    usable = []
    for entry in data:
        usable.append(re.sub(r'[^\w+\s]', ' ', entry).split())
    return usable


def buildModel(documentset, embedding):
    if embedding == 'word2vec':
        model = Word2Vec(size=DIMENSIONS, min_count=1)
    elif embedding == 'doc2vec':
        model = Doc2Vec(size=DIMENSIONS, min_count=1)
    elif embedding == 'fasttext':
        model = FastText(size=DIMENSIONS, window=5, min_count=1, workers=4)
    else:
        raise ValueError(
            'Please specify the preferred embedding to build a model! Currently only "word2vec" , "doc2vec" and "fasttext" are supported.')
    model.build_vocab(sentences=documentset)
    model.train(documentset, total_examples=model.corpus_count, epochs=model.epochs)
    return model


def get_column_embedding(col1, col2):
    documents = separate_words(col1.append(col2))
    print('Start building')
    model = FastText(documents, size=DIMENSIONS, window=5, min_count=1, workers=4)
    print('Model built')
    vecs = []
    for document in documents:
        sum_vec = Vector([0] * DIMENSIONS)
        for word in document:
            vec = Vector(model.wv[word])
            sum_vec += vec
        sum_vec = sum_vec.multiply(1 / len(document))
        vecs.append(sum_vec.vec)
    return pd.DataFrame(vecs)
