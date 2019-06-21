import numpy as np
import pandas as pd
from gensim.models import FastText, Word2Vec
from numbers import Number
from copy import copy

from gensim.models.base_any2vec import BaseWordEmbeddingsModel

from Morpheus.precalculations.precalculation import Precalculation
from Morpheus.precalculations.text_precalculation import TokenizeIntoLowerWordsPrecalculation
from Morpheus.precalculations.store import Store


class TextEmbeddingPrecalculation(Precalculation):

    def __init__(self, model=None, trained_model=None, agg=None):
        self.model = None
        self.trained_model = None
        self.agg = agg

        if trained_model:
            self.trained_model = trained_model
        elif model == 'fasttext':
            self.model = FastText(size=100, window=5, min_count=1, workers=4)
        elif model == 'word2vec':
            self.model = Word2Vec(size=100, window=5, min_count=1, workers=4)
        elif isinstance(model, BaseWordEmbeddingsModel):
            # TODO: make model's params size and window configurable through the constructor
            self.model = model
        else:
            raise ValueError('Invalid model')

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, self.__class__):
            if self.trained_model:
                return self.trained_model == other.trained_model
            else:
                if not other.model:
                    return False

                model_attributes = sorted([(k, v) for k, v in self.model.__dict__.items()
                                           if isinstance(v, Number) or isinstance(v, str)])
                other_model_attributes = sorted([(k, v) for k, v in other.model.__dict__.items()
                                                 if isinstance(v, Number) or isinstance(v, str)])

                if isinstance(other.model, self.model.__class__) \
                        and model_attributes == other_model_attributes \
                        and self.agg == other.agg:
                    return True
        return False

    def __hash__(self):
        """Overrides the default implementation"""
        if not self.model:
            return hash(self.trained_model)
        model_attributes = [(k, v) for k, v in self.model.__dict__.items()
                            if isinstance(v, Number) or isinstance(v, str)]
        return hash(tuple([self.model.__class__, self.agg] + sorted(model_attributes)))

    def process_cell(self, cell):
        ser = [self.trained_model.wv[word] for word in cell]

        if self.agg == 'sum':
            ser = np.sum(ser, axis=0)
        elif self.agg == 'avg':
            ser = np.mean(ser, axis=0)
        else:
            ser = np.array(ser)

        # return vector of zeros when cell was empty
        if self.agg is None and ser.shape == (0,):
            # create 2D vector of shape (1, vector_size)
            ser = np.array([np.array([0.0] * self.trained_model.vector_size)])
        elif self.agg is not None and type(ser) == np.float64:
            # create 1D vector of shape (vector_size)
            ser = np.array([0.0] * self.trained_model.vector_size)

        return ser

    def process(self, store: Store):
        df1_tokenized, df2_tokenized = store[TokenizeIntoLowerWordsPrecalculation()]

        concatenated_ser = pd.concat([df1_tokenized[i] for i in df1_tokenized] +
                                     [df2_tokenized[i] for i in df2_tokenized])

        if not self.trained_model:
            model = copy(self.model)
            model.build_vocab(sentences=concatenated_ser)
            model.train(sentences=concatenated_ser, total_examples=len(concatenated_ser), epochs=10)
            self.trained_model = model

        df1_embedding = pd.DataFrame()
        df2_embedding = pd.DataFrame()
        for column in df1_tokenized:
            df1_embedding[column] = df1_tokenized[column].apply(self.process_cell)
        for column in df2_tokenized:
            df2_embedding[column] = df2_tokenized[column].apply(self.process_cell)

        return df1_embedding, df2_embedding
