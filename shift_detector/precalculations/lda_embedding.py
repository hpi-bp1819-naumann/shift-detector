import pandas as pd
import numpy as np
from numbers import Number
from sklearn.decomposition import LatentDirichletAllocation as LDA_skl
from sklearn.feature_extraction.text import *
from gensim.sklearn_api import LdaTransformer
from gensim.corpora import Dictionary
import warnings
from shift_detector.precalculations.precalculation import Precalculation
from shift_detector.precalculations.count_vectorizer import CountVectorizer
from shift_detector.precalculations.lda_gensim_tokenizer import LdaGensimTokenizer
from shift_detector.utils.column_management import ColumnType


class LdaEmbedding(Precalculation):

    def __init__(self, cols, n_topics=20, n_iter=10, lib='sklearn', random_state=0, trained_model=None,
                 stop_words='english', max_features=None):
        self.model = None
        self.trained_model = None
        self.cols = None
        self.stop_words = stop_words
        self.max_features = max_features

        if n_topics != 'auto':
            # TODO implement feature to calculate optimal number of topics
            self.n_topics = n_topics
        if not isinstance(n_topics, int):
            raise TypeError("Number of topic has to be an integer. Received: {}".format(type(n_topics)))
        if n_topics < 2:
            raise ValueError("Number of topics has to be at least 2. Received: {}".format(n_topics))
        self.n_topics = n_topics

        if not isinstance(n_iter, int):
            raise TypeError("Random_state has to be a integer. Received: {}".format(type(n_iter)))
        if n_iter < 1:
            raise ValueError("Random_state has to be at least 1. Received: {}".format(n_iter))
        self.n_iter = n_iter

        if not isinstance(random_state, int):
            raise TypeError("Random_state has to be a integer. Received: {}".format(type(random_state)))
        if random_state < 0:
            raise ValueError("Random_state has to be positive or zero. Received: {}".format(random_state))
        self.random_state = random_state

        if trained_model:
            warnings.warn("Trained models are not trained again. Please make sure to only input the column(s) "
                          "that the model was trained on", UserWarning)
            self.trained_model = trained_model
        else:
            if not isinstance(lib, str):
                raise TypeError("Lib has to be a string. Received: {}".format(type(lib)))
            if lib == 'sklearn':
                self.model = \
                    LDA_skl(n_components=self.n_topics, max_iter=self.n_iter, random_state=self.random_state)
            elif lib == 'gensim':
                self.model = \
                    LdaTransformer(num_topics=self.n_topics, iterations=self.n_iter, random_state=self.random_state)
            else:
                raise ValueError("The supported libraries are sklearn and gensim. Received: {}".format(lib))
        self.lib = lib

        if cols:
            if isinstance(cols, list) and all(isinstance(col, str) for col in cols):
                self.cols = cols
            else:
                raise TypeError("Cols has to be list of strings . Column {} is of type {}".format(cols, type(cols)))
        else:
            raise ValueError("You have to specify which columns you want to vectorize")

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            model_attributes = sorted([(k, v) for k, v in self.model.__dict__.items()
                                       if isinstance(v, Number) or isinstance(v, str) or isinstance(v, list)])
            other_model_attributes = sorted([(k, v) for k, v in other.model.__dict__.items()
                                             if isinstance(v, Number) or isinstance(v, str) or isinstance(v, list)])
            return isinstance(other.model, self.model.__class__) \
                and model_attributes == other_model_attributes and self.cols == other.cols \
                and self.stop_words == other.stop_words and self.max_features == other.max_features
        return False

    def __hash__(self):
        if self.trained_model:
            trained_hash_list = [self.__class__, self.trained_model.__class__]
            for item in self.trained_model.__dict__.items():
                if not item[0] == 'components_' and not item[0] == 'exp_dirichlet_component_':
                    # dirty fix I know, ndarrays are not hashable
                    trained_hash_list.extend(item)
            return hash(tuple(trained_hash_list))
        elif self.cols:
            hash_list = [self.__class__, self.model.__class__, self.n_topics,
                         self.n_iter, self.random_state, self.max_features]
            hash_list.extend(self.cols)
            hash_list.extend(self.stop_words)
            return hash(tuple(hash_list))
        else:
            return hash(tuple([self.__class__, self.model.__class__, self.n_topics,
                               self.n_iter, self.random_state]))

    @staticmethod
    def topic_probabilities_to_topics(lda_model, dtm):
        # Always takes the topic with the highest probability as the dominant topic
        return [arr.argmax()+1 for arr in lda_model.transform(dtm)]

    @staticmethod
    def get_topic_word_distribution_gensim(lda_model, n_topics, n_top_words):
        topic_words = lda_model.gensim_model.show_topics(num_topics=n_topics,
                                                         num_words=n_top_words,
                                                         formatted=False)
        return topic_words

    @staticmethod
    def get_topic_word_distribution_sklearn(lda_model, vocab, n_top_words):
        # copied implementation from gensim show_topics
        topic_words = []
        for topic_n, comp in enumerate(lda_model.components_):
            topic_ = comp
            topic_ = topic_ / topic_.sum()
            most_extreme = np.argpartition(-topic_, n_top_words)[:n_top_words]
            word_idx = most_extreme.take(np.argsort(-topic_.take(most_extreme)))
            topic_ = [(vocab[id], topic_[id]) for id in word_idx]
            topic_words.append((topic_n, topic_))
        return topic_words

    def process(self, store):
        if isinstance(self.cols, str):
            if self.cols in store.column_names(ColumnType.text):
                col_names = self.cols
        else:
            for col in self.cols:
                if col not in store.column_names(ColumnType.text):
                    raise ValueError("Given column is not contained in detected text columns of the datasets: {}"
                                     .format(col))
            col_names = self.cols

        topic_labels = ['topics ' + col for col in col_names]

        transformed1 = pd.DataFrame()
        transformed2 = pd.DataFrame()
        topic_words_all_cols = {}
        all_models = {}

        if self.lib == 'gensim':
            tokenized1, tokenized2 = store[LdaGensimTokenizer(stop_words=self.stop_words, cols=self.cols)]
            tokenized_merged = pd.concat([tokenized1, tokenized2], ignore_index=True)

            all_corpora = {}
            all_dicts = {}

            for i, col in enumerate(col_names):
                all_dicts[col] = Dictionary(tokenized_merged[col])
                gensim_dict1 = Dictionary(tokenized1[col])
                gensim_dict2 = Dictionary(tokenized2[col])

                all_corpora[col] = [all_dicts[col].doc2bow(line) for line in tokenized_merged[col]]
                corpus1 = [gensim_dict1.doc2bow(line) for line in tokenized1[col]]
                corpus2 = [gensim_dict2.doc2bow(line) for line in tokenized2[col]]

                if not self.trained_model:
                    model = self.model
                    model.id2word = all_dicts[col]
                    model = model.fit(all_corpora[col])
                    all_models[col] = model.gensim_model
                else:
                    model = self.trained_model

                topic_words_all_cols[col] = self.get_topic_word_distribution_gensim(model, self.n_topics, 200)

                transformed1[topic_labels[i]] = self.topic_probabilities_to_topics(model, corpus1)
                transformed2[topic_labels[i]] = self.topic_probabilities_to_topics(model, corpus2)

            return transformed1, transformed2, topic_words_all_cols, all_models, all_corpora, all_dicts

        else:
            vectorized1, vectorized2, feature_names, all_vecs = store[CountVectorizer(stop_words=self.stop_words,
                                                                                      max_features=self.max_features,
                                                                                      cols=self.cols)]
            all_dtms = dict(vectorized1, **vectorized2)

            for i, col in enumerate(col_names):
                if not self.trained_model:
                    model = self.model
                    model = model.fit(all_dtms[col])
                    all_models[col] = model
                else:
                    model = self.trained_model

                topic_words_all_cols[col] = self.get_topic_word_distribution_sklearn(model,
                                                                                     feature_names[col],
                                                                                     200)

                transformed1[topic_labels[i]] = \
                    self.topic_probabilities_to_topics(model, vectorized1[col])
                transformed2[topic_labels[i]] = \
                    self.topic_probabilities_to_topics(model, vectorized2[col])

            return transformed1, transformed2, topic_words_all_cols, all_models, all_dtms, all_vecs
