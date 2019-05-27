import numpy as np
import gensim
from copy import copy
import pandas as pd
from shift_detector.preprocessors.Preprocessor import Preprocessor
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer as CountVectorizer_sklearn


class CountVectorizer(Preprocessor):

    def __init__(self, stop_words='english'):
        self.stopwords = None
        if isinstance(stop_words, str):
            self.stop_words = stopwords.words(stop_words)
        elif isinstance(stop_words, list):
            if all(isinstance(elem, str) for elem in stop_words):
                for lang in stop_words:
                    if self.stopwords is None:
                        self.stopwords = set(stopwords.words(lang))
                    else:
                        self.stopwords = self.stopwords.union(set(stopwords.words(lang)))
        else:
            raise Exception('Please enter the language for your stopwords as a string or list of strings')
    
    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, self.__class__):
            if self.stop_words == other.stop_words:
                return True
        return False

    def __hash__(self):
        """Overrides the default implementation"""
        return hash(tuple([self.__class__] + sorted(self.stop_words)))

    def process(self, store):
        train_texts, test_texts = store[ColumnType.text]
        merged_texts = pd.concat([train_texts, test_texts])
        vectorized_merged = CountVectorizer_sklearn().fit_transform(merged_texts)
        vectorized_train = CountVectorizer_sklearn().fit_transform(train_texts)
        vectorized_test = CountVectorizer_sklearn().fit_transform(test_texts)
        return processed
