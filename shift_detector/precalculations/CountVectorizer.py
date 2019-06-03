import pandas as pd
from shift_detector.precalculations.Precalculation import Precalculation
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer as CountVectorizer_sklearn
from shift_detector.Utils import ColumnType



class CountVectorizer(Precalculation):

    def __init__(self, stop_words='english', max_features=1000):
        self.stopwords = None
        self.max_features = None
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
        if isinstance(max_features, int):
            if max_features > 0:
                self.max_features = max_features
            else:
                raise ValueError('Max_features has to be at least 1')
        else:
            raise Exception('Max_features has to be an int')
        self.vectorizer = CountVectorizer_sklearn(stop_words=self.stopwords, max_features=self.max_features)


    
    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, self.__class__) and self.stop_words == other.stop_words \
                and self.max_features == other.max_features:
                    return True
        return False

    def __hash__(self):
        """Overrides the default implementation"""
        # TODO investigate why the error 'unhashable type list' is thrown here
        return hash(tuple([self.__class__, self.stop_words, self.max_features]))

    def process(self, store):
        train_texts, test_texts = store[ColumnType.text]
        merged_texts = pd.concat([train_texts, test_texts], ignore_index=True)
        self.vectorizer = self.vectorizer.fit(merged_texts)
        vectorized_merged = self.vectorizer.transform(merged_texts)
        vectorized_train = CountVectorizer_sklearn().transform(train_texts)
        vectorized_test = CountVectorizer_sklearn().transform(test_texts)
        return vectorized_merged, vectorized_train, vectorized_test
