import pandas as pd
from shift_detector.precalculations.Precalculation import Precalculation
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer as CountVectorizer_sklearn
from shift_detector.Utils import ColumnType



class CountVectorizer(Precalculation):

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
        return vectorized_merged, vectorized_train, vectorized_test
