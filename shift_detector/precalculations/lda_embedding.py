import pandas as pd
from numbers import Number
from sklearn.decomposition import LatentDirichletAllocation as LDA_skl
from sklearn.feature_extraction.text import *
from gensim.sklearn_api import LdaTransformer
from gensim.corpora import Dictionary
import lda
from copy import copy
from shift_detector.precalculations.precalculation import Precalculation
from shift_detector.precalculations.count_vectorizer import CountVectorizer
from shift_detector.precalculations.word_tokenizer import WordTokenizer
from shift_detector.utils.column_management import ColumnType


class LdaEmbedding(Precalculation):

    def __init__(self, cols, n_topics=20, n_iter=10, lib='sklearn', random_state=0, trained_model=None,
                 stop_words='english', max_features=None):
        self.model = None
        self.trained_model = None
        self.cols = None
        if n_topics < 2:
            raise ValueError('Number of topics has to be at least 2')
        if n_topics != 'auto':
            self.n_topics = n_topics
        self.n_iter = n_iter
        if lib in ['sklearn', 'gensim', 'lda']:
            self.lib = lib
        else:
            raise ValueError('The supported libraries are sklearn, gensim and lda')
        if isinstance(random_state, int):
            if random_state >= 0:
                self.random_state = random_state
            else:
                raise ValueError('Random_state has to be positive')
        else:
            raise TypeError('Random_state has to be a integer')
        if trained_model:
            self.trained_model = trained_model
        elif lib == 'sklearn':
            self.model = LDA_skl(n_components=self.n_topics, max_iter=self.n_iter, random_state=self.random_state)
        elif lib == 'gensim':
            self.model = \
                LdaTransformer(num_topics=self.n_topics, iterations=self.n_iter, random_state=self.random_state)
        elif lib == 'lda':
            self.model = lda.LDA(n_topics=self.n_topics, n_iter=self.n_iter, random_state=self.random_state)
            # n_iter is only the amount of sample iterations, so it can be much higher than the iterations parameter of
            # the other models without sacrificing performance
        else:
            raise ValueError('Please enter one of the supported libraries: "sklearn", "gensim" or "lda"')
        if cols is not None:
            if isinstance(cols, list) and all(isinstance(col, str) for col in cols) or isinstance(cols, str):
                self.cols = cols
            else:
                raise TypeError('Cols has to be list of strings or a single string')
        else:
            raise ValueError('You have to specify which columns you want to vectorize')
        self.stop_words = stop_words
        self.max_features = max_features
    
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            model_attributes = sorted([(k, v) for k, v in self.model.__dict__.items()
                                       if isinstance(v, Number) or isinstance(v, str) or isinstance(v, list)])
            other_model_attributes = sorted([(k, v) for k, v in other.model.__dict__.items()
                                             if isinstance(v, Number) or isinstance(v, str) or isinstance(v, list)])
            if isinstance(other.model, self.model.__class__) \
                    and model_attributes == other_model_attributes and self.cols == other.cols \
                    and self.stop_words == other.stop_words and self.max_features == other.max_features:
                return True
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

    def process(self, store):

        df1_texts, df2_texts = store[ColumnType.text]
        if self.cols is None:
            col_names = df1_texts.columns
            self.cols = col_names
        else:
            if isinstance(self.cols, str):
                if self.cols in df1_texts.columns:
                    col_names = self.cols
            else:
                for col in self.cols:
                    if col not in df1_texts.columns:
                        raise ValueError('Given column is not contained in given datasets')
                col_names = self.cols

        topic_labels = []
        for col in col_names:
            topic_labels.append('topics ' + col)

        transformed1 = {}
        transformed2 = {}

        topics1 = pd.DataFrame(index=df1_texts.index, columns=topic_labels)
        topics2 = pd.DataFrame(index=df2_texts.index, columns=topic_labels)

        if self.lib == 'gensim':
            tokenized1, tokenized2 = store[WordTokenizer(stop_words=self.stop_words, cols=self.cols)]
            tokenized_merged = pd.concat([tokenized1, tokenized2], ignore_index=True)

            for col in col_names:
                dict_merged = Dictionary(tokenized_merged[col])
                dict1 = Dictionary(tokenized1[col])
                dict2 = Dictionary(tokenized2[col])

                corpus_merged = [dict_merged.doc2bow(line) for line in tokenized_merged[col]]
                corpus1 = [dict1.doc2bow(line) for line in tokenized1[col]]
                corpus2 = [dict2.doc2bow(line) for line in tokenized2[col]]

                if not self.trained_model:
                    model = copy(self.model)
                    model = model.fit(corpus_merged)
                    self.trained_model = model

                transformed1[col] = self.trained_model.transform(corpus1)
                transformed2[col] = self.trained_model.transform(corpus2)

        else:
            vectorized1, vectorized2 = store[CountVectorizer(stop_words=self.stop_words, max_features=self.max_features,
                                                             cols=self.cols)]
            vectorized_merged = dict(vectorized1, **vectorized2)

            for col in col_names:
                if not self.trained_model:
                    model = copy(self.model)
                    model = model.fit(vectorized_merged[col])
                    self.trained_model = model
                transformed1[col] = self.trained_model.transform(vectorized1[col])
                transformed2[col] = self.trained_model.transform(vectorized2[col])

        for i, col in enumerate(col_names):
            vec1 = [0] * len(transformed1[col])
            vec2 = [0] * len(transformed2[col])
            # infer topics for train_df
            for j in range(len(transformed1[col])):
                # take always the topic with the highest probability
                vec1[j] = transformed1[col][j].argmax()
            topics1[topic_labels[i]] = vec1
            # infer topics for test_df
            for k in range(len(transformed2[col])):
                # take always the topic with the highest probability
                vec2[k] = transformed2[col][k].argmax()
            topics2[topic_labels[i]] = vec2

        return topics1, topics2
