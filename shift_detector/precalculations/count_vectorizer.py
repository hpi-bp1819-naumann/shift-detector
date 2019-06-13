import pandas as pd
from shift_detector.precalculations.precalculation import Precalculation
from nltk.corpus import stopwords as nltk_stopwords
from sklearn.feature_extraction.text import CountVectorizer as CountVectorizer_sklearn
from shift_detector.utils.column_management import ColumnType


class CountVectorizer(Precalculation):

    def __init__(self, cols, stop_words='english', max_features=None):
        # potentially make min_df and max_df available
        self.stopwords = None
        self.max_features = None
        self.cols = None
        if isinstance(stop_words, str):
            if stop_words in nltk_stopwords.fileids():
                self.stop_words = nltk_stopwords.words(stop_words)
            else:
                raise Exception('The language you entered is not available')
        elif isinstance(stop_words, list):
            if all(isinstance(elem, str) for elem in stop_words):
                for lang in stop_words:
                    if lang not in nltk_stopwords.fileids():
                        raise Exception('At least one language you entered is not available')
                    if self.stopwords is None:
                        self.stopwords = set(nltk_stopwords.words(lang))
                    else:
                        self.stopwords = self.stopwords.union(set(nltk_stopwords.words(lang)))
            else:
                raise TypeError('The stop_words list has to contain strings only')
        else:
            raise TypeError('Please enter the language for your stopwords as a string or list of strings')
        if max_features is not None:
            if isinstance(max_features, int):
                if max_features > 0:
                    self.max_features = max_features
                else:
                    raise ValueError('Max_features has to be at least 1')
            else:
                raise TypeError('Max_features has to be an int')
        if cols is not None:
            if isinstance(cols, list) and all(isinstance(col, str) for col in cols) or isinstance(cols, str):
                self.cols = cols
            else:
                raise TypeError('Cols has to be list of strings or a single string')
        else:
            raise ValueError('You have to specify which columns you want to vectorize')

        self.vectorizer = CountVectorizer_sklearn(stop_words=self.stopwords, max_features=self.max_features)

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, self.__class__) and self.stop_words == other.stop_words \
                and self.max_features == other.max_features and self.cols == other.cols:
            return True
        return False

    def __hash__(self):
        """Overrides the default implementation"""
        hash_list = [self.__class__, self.max_features]
        hash_list.extend(sorted(self.stop_words))
        hash_list.extend(self.cols)
        return hash(tuple(hash_list))

    def process(self, store):
        df1_texts, df2_texts = store[ColumnType.text]
        merged_texts = pd.concat([df1_texts, df2_texts], ignore_index=True)

        if isinstance(self.cols, str):
            if self.cols in df1_texts.columns:
                col_names = [self.cols]
        else:
            for col in self.cols:
                if col not in df1_texts.columns:
                    raise ValueError('Given column is not contained in given datasets')
            col_names = self.cols

        dict_of_sparse_matrices1 = {}
        dict_of_sparse_matrices2 = {}

        for col in col_names:
            count_vec = self.vectorizer
            count_vec = count_vec.fit(merged_texts[col])
            dict_of_sparse_matrices1[col] = count_vec.transform(df1_texts[col]).A.astype(int)
            # you can also leave the last part and you get a sparse matrix that is much more memory efficient
            dict_of_sparse_matrices2[col] = count_vec.transform(df2_texts[col]).A.astype(int)
        return dict_of_sparse_matrices1, dict_of_sparse_matrices2
