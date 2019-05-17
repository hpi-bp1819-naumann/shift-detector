import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation as LDA_skl
from gensim.models import LdaMulticore
from gensim.corpora import Dictionary
import lda
from copy import copy
from shift_detector.preprocessors.Preprocessor import Preprocessor
import nltk

class LDA(Preprocessor):

    def __init__(self, n_topics=20, type=None, trained_model=None):
        if n_topics != 'auto':
            self.n_topics = n_topics
        self.model = None
        self.trained_model = None
        self.type = type
        if trained_model:
            self.trained_model = trained_model
        elif type == 'sklearn':
            self.model = LDA_skl(n_components=self.n_topics, random_state=0)
        elif type == 'gensim':
            self.model = LdaMulticore(num_topics=self.n_topics, random_state=0)
        elif type == 'lda':
            self.model = lda.LDA(n_topics=self.n_topics, random_state=0)
        else:
            raise Exception('No LDA type defined')
    
    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, self.__class__):
            if self.trained_model and self.trained_model == other.trained_model \
                    or self.model == other.model:
                return True

    def __hash__(self):
        """Overrides the default implementation"""
        if not self.model:
            return hash(self.trained_model)
        return hash(tuple([self.model.__class__] + self.n_topics))

    def process(self, train_df, test_df):

        if not self.trained_model:
            model = copy(self.model)
            df_merged = pd.concat([train_df, test_df])

            if self.type == 'sklearn':
                # TODO use Store here
                tokenized = CountVectorizer().fit_transform(df_merged)
                self.model.fit(tokenized)
            elif self.type == 'gensim':
                # TODO use Store here, also have to get the right text column(s)
                word_tokenized = nltk.word_tokenize(df_merged)
                train_word_tokenized = nltk.word_tokenize(train_df)
                test_word_tokenized = nltk.word_tokenize(test_df)
                df_dict = Dictionary(word_tokenized)
                df_corpus = [df_dict.doc2bow(line) for line in word_tokenized]
                train_dict = Dictionary(train_word_tokenized)
                train_corpus = [train_dict.doc2bow(line) for line in train_word_tokenized]
                test_dict = Dictionary(test_word_tokenized)
                test_corpus = [test_dict.doc2bow(line) for line in test_word_tokenized]
                self.model = self.model(df_corpus, id2word=df_dict, num_topics=self.n_topics, random_state=0)

                # infer topics for train_df
                inferred_train_vec = len(train_corpus) * [0]
                for i in range(len(train_corpus)):
                    inferred_train_vec[i] = sorted(self.model[train_corpus[i]],
                                                    key=lambda tup: tup[1],
                                                    reverse=True)[:1][0][0]

                # infer topics for test_df
                inferred_test_vec = len(test_corpus) * [0]
                for i in range(len(test_corpus)):
                    inferred_test_vec[i] = sorted(self.model[test_corpus[i]],
                                                  key=lambda tup: tup[1],
                                                  reverse=True)[:1][0][0]

                train_df['topic'] = inferred_train_vec
                test_df['topic'] = inferred_test_vec
            #model.build_vocab(sentences=df)
            #model.train(sentences=df, total_examples=len(df), epochs=1)
            #self.trained_model = model

        #processed1 = df1.apply(lambda row: np.sum([self.trained_model.wv[word] for word in row], axis=0))
        #processed2 = df2.apply(lambda row: np.sum([self.trained_model.wv[word] for word in row], axis=0))

        return train_df, test_df
