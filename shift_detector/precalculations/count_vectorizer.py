import pandas as pd
from shift_detector.precalculations.precalculation import Precalculation
from nltk.corpus import stopwords as nltk_stopwords
from sklearn.feature_extraction.text import CountVectorizer as CountVectorizer_sklearn
from shift_detector.utils.column_management import ColumnType
from bs4 import BeautifulSoup


class CountVectorizer(Precalculation):

    def __init__(self, columns, stop_words='english', max_features=None):
        # potentially make min_df and max_df available
        self.max_features = None

        if isinstance(stop_words, str):
            if stop_words not in nltk_stopwords.fileids():
                raise ValueError("The language you entered is not available. Received: {}".format(stop_words))
            self.stop_words = set(nltk_stopwords.words(stop_words))
        elif isinstance(stop_words, list) and all(isinstance(elem, str) for elem in stop_words):
            self.stop_words = set()
            for lang in stop_words:
                if lang not in nltk_stopwords.fileids():
                    raise ValueError("The following language you entered is not available: {}".format(lang))
                self.stop_words = self.stop_words.union(set(nltk_stopwords.words(lang)))
        else:
            raise TypeError("Please enter the language for your stop_words as a string or list of strings. Received: {}"
                            .format(type(stop_words)))

        if max_features is not None:
            if not isinstance(max_features, int):
                raise TypeError("Max_features has to be a integer. Received: {}".format(type(max_features)))
            if max_features < 1:
                raise ValueError("Max_features has to be at least 1. Received: {}".format(max_features))
            self.max_features = max_features

        if not columns:
            raise TypeError("You have to specify which columns you want to tokenize")
        if isinstance(columns, list) and all(isinstance(col, str) for col in columns):
            self.columns = columns
        else:
            raise TypeError("Columns has to be list of strings or a single string. Received: {}".format(type(columns)))

        self.vectorizer = CountVectorizer_sklearn(stop_words=self.stop_words,
                                                  max_features=self.max_features,
                                                  max_df=0.5)

    def __eq__(self, other):
        """Overrides the default implementation"""
        return isinstance(other, self.__class__) and self.stop_words == other.stop_words \
            and self.max_features == other.max_features and self.columns == other.columns

    def __hash__(self):
        """Overrides the default implementation"""
        hash_list = [self.__class__, self.max_features]
        hash_list.extend(sorted(self.stop_words))
        hash_list.extend(self.columns)
        return hash(tuple(hash_list))

    def process(self, store):
        df1_texts, df2_texts = store[ColumnType.text]
        merged_texts = pd.concat([df1_texts, df2_texts], ignore_index=True)
        # Remove HTML tags
        cleaned_texts = merged_texts.applymap(lambda text: BeautifulSoup(text, features="lxml").get_text())

        for col in self.columns:
            if col not in store.column_names(ColumnType.text):
                raise ValueError("Given column is not contained in detected text columns of the datasets: {}"
                                 .format(col))
        col_names = self.columns

        dict_of_arrays1 = {}
        dict_of_arrays2 = {}
        feature_names = {}
        all_vecs = {}

        for col in col_names:
            all_vecs[col] = self.vectorizer
            all_vecs[col] = all_vecs[col].fit(cleaned_texts[col])
            feature_names[col] = all_vecs[col].get_feature_names()
            dict_of_arrays1[col] = all_vecs[col].transform(df1_texts[col]).toarray().astype(int)
            dict_of_arrays2[col] = all_vecs[col].transform(df2_texts[col]).toarray().astype(int)

        return dict_of_arrays1, dict_of_arrays2, feature_names, all_vecs
