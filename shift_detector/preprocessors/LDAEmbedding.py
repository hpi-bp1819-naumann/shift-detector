import numpy as np
import pandas as pd
from numbers import Number
from sklearn.decomposition import LatentDirichletAllocation as LDA_skl
from sklearn.feature_extraction.text import *
from gensim.sklearn_api import LdaTransformer
from gensim.corpora import Dictionary
import lda
from copy import copy
from shift_detector.preprocessors.Preprocessor import Preprocessor
from shift_detector.preprocessors.WordTokenizer import WordTokenizer
from shift_detector.Utils import ColumnType


class LDAEmbedding(Preprocessor):

    def __init__(self, n_topics=20, n_iter=100, lib=None, trained_model=None):
        if n_topics != 'auto':
            self.n_topics = n_topics
        self.n_iter = n_iter
        self.model = None
        self.trained_model = None
        self.lib = lib
        if trained_model:
            self.trained_model = trained_model
        elif lib == 'sklearn':
            self.model = LDA_skl(n_components=self.n_topics, max_iter=self.n_iter, random_state=0, n_jobs: 0)
        elif lib == 'gensim':
            self.model = LdaTransformer(num_topics=self.n_topics, iterations=self.n_iter, random_state=0)
        elif lib == 'lda':
            self.model = lda.LDA(n_topics=self.n_topics, n_iter=self.n_iter, random_state=0)
            # n_iter is only the amount of sample iterations, so it can be much higher than the iterations parameter of
            # the other models without sacrificing performance
        else:
            raise Exception('No LDA library defined')
    
    def __eq__(self, other):
        """Overrides the default implementation"""
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
        """Overrides the default implementation"""
        if not self.model:
            return hash(self.trained_model)
        return hash(tuple([self.model.__class__] + self.n_topics))

    def process(self, store):


        inferred_train_vec = train_df.shape[0] * [0]
        inferred_test_vec = test_df.shape[0] * [0]

        if self.lib == 'gensim':
            # TODO do preprocessing in different class with store
            merged_tokenized, train_tokenized, test_tokenized = store[WordTokenizer()]
            merged_dict = Dictionary(merged_tokenized)
            train_dict = Dictionary(train_tokenized)
            test_dict = Dictionary(test_tokenized)
            merged_corpus = [merged_dict.doc2bow(line) for line in merged_tokenized]
            train_corpus = [train_dict.doc2bow(line) for line in train_tokenized]
            test_corpus = [test_dict.doc2bow(line) for line in test_tokenized]
            if not self.trained_model:
                model = copy(self.model)
                model = model.fit(merged_corpus)
                self.trained_model = model
            transformed_train = self.trained_model.transform(train_corpus)
            transformed_test = self.trained_model.transform(test_corpus)

        else:
            # TODO do preprocessing in different class with store
            train_texts, test_texts = store[ColumnType.text]
            merged_texts = pd.concat([train_texts, test_texts])

            vectorized_merged = CountVectorizer().fit_transform(merged_texts)
            vectorized_train = CountVectorizer().fit_transform(train_texts)
            vectorized_test = CountVectorizer().fit_transform(test_texts)

            if not self.trained_model:
                model = copy(self.model)
                model = model.fit(vectorized_merged)
                self.trained_model = model
            transformed_train = self.trained_model.transform(vectorized_train)
            transformed_test = self.trained_model.transform(vectorized_test)

        # infer topics for train_df
        for i in range(len(transformed_train)):
            inferred_train_vec[i] = transformed_train[i].argmax()

        # infer topics for test_df
        for i in range(len(transformed_test)):
            inferred_test_vec[i] = transformed_test[i].argmax()

        train_df['topic'] = inferred_train_vec
        test_df['topic'] = inferred_test_vec

        return train_df, test_df
