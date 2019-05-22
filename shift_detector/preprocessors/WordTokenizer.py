import numpy as np
import gensim
from copy import copy
from shift_detector.preprocessors.Preprocessor import Preprocessor
from nltk.corpus import stopwords


class WordTokenizer(Preprocessor):

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
            if self.model == other.model:
                return True
            if isinstance(other.model, self.model.__class__) \
                    and self.model.stop_words == other.model.stop_words:
                return True
        return False

    def __hash__(self):
        """Overrides the default implementation"""
        return hash(tuple([self.model.__class__] + sorted(self.model.stop_words)))

    def remove_stopwords(self, texts):
        return [[word for word in gensim.utils.simple_preprocess(str(doc), deacc=True) if word not in self.stopwords]
                for doc in texts]

    def process(self, store):
        # TODO: do everything in the process function
        processed = remove_stopwords([gensim.utils.simple_preprocess(text, deacc=True) for text in data])

        return processed
