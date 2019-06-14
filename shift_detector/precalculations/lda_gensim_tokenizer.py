import gensim
import pandas as pd
from shift_detector.precalculations.precalculation import Precalculation
from shift_detector.utils.column_management import ColumnType
from nltk.corpus import stopwords as nltk_stopwords
import re


class LdaGensimTokenizer(Precalculation):

    def __init__(self, cols, stop_words='english'):
        self.stopwords = None
        self.cols = None
        if isinstance(stop_words, str):
            if stop_words in nltk_stopwords.fileids():
                self.stop_words = nltk_stopwords.words(stop_words)
            else:
                raise ValueError('The language you entered is not available')
        elif isinstance(stop_words, list) and all(isinstance(elem, str) for elem in stop_words):
                for lang in stop_words:
                    if lang not in nltk_stopwords.fileids():
                        raise ValueError('At least one language you entered is not available')
                    if self.stopwords is None:
                        self.stopwords = set(nltk_stopwords.words(lang))
                    else:
                        self.stopwords = self.stopwords.union(set(nltk_stopwords.words(lang)))
        else:
            raise TypeError('Please enter the language for your stopwords as a string or list of strings')

        if cols:
            if isinstance(cols, list) and all(isinstance(col, str) for col in cols) or isinstance(cols, str):
                self.cols = cols
            else:
                raise TypeError('Cols has to be list of strings or a single string')
        else:
            raise TypeError('You have to specify which columns you want to tokenize')

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, self.__class__) and self.stop_words == other.stop_words and self.cols == other.cols:
            return True
        return False

    def __hash__(self):
        """Overrides the default implementation"""
        hash_list = [self.__class__]
        hash_list.extend(sorted(self.stop_words))
        hash_list.extend(self.cols)
        return hash(tuple(hash_list))

    def process(self, store):
        df1_texts, df2_texts = store[ColumnType.text]

        if isinstance(self.cols, str):
            if self.cols in df1_texts.columns:
                col_names = [self.cols]
        else:
            for col in self.cols:
                if col not in df1_texts.columns:
                    raise ValueError('Given column is not contained in given datasets')
            col_names = self.cols

        processed1 = {}
        processed2 = {}

        for col in col_names:
            tokenized1 = []
            tokenized2 = []

            for entry in df1_texts[col]:
                wordlist = []
                for word in re.sub(r"[^\w+\s]|\b[a-zA-Z]\b", ' ', entry).split():
                    if word.lower() not in self.stop_words or '':
                        if len(gensim.utils.simple_preprocess(word, deacc=True)) > 0:
                            wordlist.append(gensim.utils.simple_preprocess(word, deacc=True).pop())
                tokenized1.append(wordlist)

            processed1[col] = tokenized1

            for entry in df2_texts[col]:
                wordlist = []
                for word in re.sub(r"[^\w+\s]|\b[a-zA-Z]\b", ' ', entry).split():
                    if word.lower() not in self.stop_words or '':
                        if len(gensim.utils.simple_preprocess(word, deacc=True)) > 0:
                            wordlist.append(gensim.utils.simple_preprocess(word, deacc=True).pop())
                tokenized2.append(wordlist)

            processed2[col] = tokenized2

        return pd.DataFrame.from_dict(processed1), pd.DataFrame.from_dict(processed2)
