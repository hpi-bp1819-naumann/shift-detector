from enum import Enum
import numpy as np
import gensim
from copy import copy
from shift_detector.preprocessors.Preprocessor import Preprocessor


class WordTokenizer(Preprocessor):

    def __init__(self, model=None, trained_model=None):
        self.model = None
        self.trained_model = None

        if trained_model:
            self.trained_model = trained_model
        elif not isinstance(model):
            self.model = model
        else:
            raise Exception('No embedding defined')
    
    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, self.__class__):
            if self.trained_model and self.trained_model == other.trained_model \
                    or self.model == other.model:
                return True

            model_attributes = sorted([(k, v) for k, v in self.model.__dict__.items()
                                       if isinstance(v, Number) or isinstance(v, str)])
            other_model_attributes = sorted([(k, v) for k, v in other.model.__dict__.items()
                                             if isinstance(v, Number) or isinstance(v, str)])
            if isinstance(other.model, self.model.__class__) \
                    and model_attributes == other_model_attributes:
                return True
        return False

    def __hash__(self):
        """Overrides the default implementation"""
        if not self.model:
            return hash(self.trained_model)
        model_attributes = [(k, v) for k, v in self.model.__dict__.items() \
                            if isinstance(v, Number) or isinstance(v, str)]
        return hash(tuple([self.model.__class__] + sorted(model_attributes)))

    def remove_stopwords(texts):
        return [[word for word in gensim.utils.simple_preprocess(str(doc), deacc=True) if word not in stop_words] for
                doc in texts]

    def process(self, store):
        processed = remove_stopwords([gensim.utils.simple_preprocess(text, deacc=True) for text in data])
        if not self.trained_model:
            model = copy(self.model)
            model.build_vocab(sentences=df)
            model.train(sentences=df, total_examples=len(df), epochs=1)
            self.trained_model = model

        processed = df.apply(lambda row: np.sum([self.trained_model.wv[word] for word in row], axis=0))
        return processed
