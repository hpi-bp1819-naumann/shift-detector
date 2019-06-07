import pandas as pd
from numbers import Number
from sklearn.decomposition import LatentDirichletAllocation as LDA_skl
from sklearn.feature_extraction.text import *
#from gensim.sklearn_api import LdaTransformer
#from gensim.corpora import Dictionary
import lda
from copy import copy
from shift_detector.precalculations.Precalculation import Precalculation
from shift_detector.precalculations.CountVectorizer import CountVectorizer
#from shift_detector.precalculations.WordTokenizer import WordTokenizer

from shift_detector.Utils import ColumnType


class LdaEmbedding(Precalculation):

    def __init__(self, n_topics=20, n_iter=10, lib='sklearn', random_state=0, trained_model=None):
        self.model = None
        self.trained_model = None
        if n_topics != 'auto':
            self.n_topics = n_topics
        if n_topics < 2:
            raise ValueError('Number of topics has to be at least 2')
        self.n_iter = n_iter
        self.lib = lib
        self.random_state = random_state
        if trained_model:
            self.trained_model = trained_model
        elif lib == 'sklearn':
            self.model = LDA_skl(n_components=self.n_topics, max_iter=self.n_iter, random_state=self.random_state)
        #elif lib == 'gensim':
         #   self.model = \
          #      LdaTransformer(num_topics=self.n_topics, iterations=self.n_iter, random_state=self.random_state)
        elif lib == 'lda':
            self.model = lda.LDA(n_topics=self.n_topics, n_iter=self.n_iter, random_state=self.random_state)
            # n_iter is only the amount of sample iterations, so it can be much higher than the iterations parameter of
            # the other models without sacrificing performance
        else:
            raise Exception('No LDA library defined')
    
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            model_attributes = sorted([(k, v) for k, v in self.model.__dict__.items()
                                       if isinstance(v, Number) or isinstance(v, str)])
            other_model_attributes = sorted([(k, v) for k, v in other.model.__dict__.items()
                                             if isinstance(v, Number) or isinstance(v, str)])
            if isinstance(other.model, self.model.__class__) \
                    and model_attributes == other_model_attributes:
                return True
        return False

    def __hash__(self):
        if self.trained_model:
            return hash(([self.__class__, self.trained_model].extend(self.trained_model.__dict__.items())))
        else:
            return hash((self.__class__, self.model.__class__, self.n_topics, self.n_iter, self.lib, self.random_state))

    def process(self, store):

        df1_texts, df2_texts = store[ColumnType.text]
        col_names = df1_texts.columns

        transformed1 = {}
        transformed2 = {}

        topics1 = pd.DataFrame()
        topics2 = pd.DataFrame()

        inferred_vec1 = {}
        inferred_vec2 = {}

        '''
        if self.lib == 'gensim':
            tokenized1, tokenized2 = store[WordTokenizer()]

            for col in col_names:

                tokenized_merged = tokenized1[col] + tokenized2[col]

                dict_merged = Dictionary(tokenized_merged)
                dict1 = Dictionary(tokenized1[col])
                dict2 = Dictionary(tokenized2[col])

                corpus_merged = [dict_merged.doc2bow(line) for line in tokenized_merged]
                corpus1 = [dict1.doc2bow(line) for line in tokenized1[col]]
                corpus2 = [dict2.doc2bow(line) for line in tokenized2[col]]

                if not self.trained_model:
                    model = copy(self.model)
                    model = model.fit(corpus_merged)
                    self.trained_model = model

                transformed1[col] = self.trained_model.transform(corpus1)
                transformed2[col] = self.trained_model.transform(corpus2)

        else:
        '''

        vectorized_train, vectorized_test = store[CountVectorizer()]
        vectorized_merged = pd.concat([vectorized_train, vectorized_test], ignore_index=True)

        for col in col_names:
            if not self.trained_model:
                model = copy(self.model)
                model = model.fit(vectorized_merged[col])
                self.trained_model = model
            transformed1[col] = self.trained_model.transform(vectorized_train[col])
            transformed2[col] = self.trained_model.transform(vectorized_test[col])

        for col in col_names:
            # infer topics for train_df
            for i in range(len(transformed1[col])):
                # take always the topic with the highest probability
                inferred_vec1[col][i] = transformed1[col][i].argmax()

            # infer topics for test_df
            for i in range(len(transformed2[col])):
                # take always the topic with the highest probability
                inferred_vec2[col][i] = transformed2[col][i].argmax()

            topics1['topics ' + col] = inferred_vec1[col]
            topics2['topics ' + col] = inferred_vec2[col]

        return topics1, topics2
