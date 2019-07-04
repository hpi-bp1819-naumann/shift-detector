import gensim
import pandas as pd
from shift_detector.precalculations.precalculation import Precalculation
from shift_detector.utils.column_management import ColumnType
from nltk.corpus import stopwords as nltk_stopwords
import re
from bs4 import BeautifulSoup


class LdaGensimTokenizer(Precalculation):

    def __init__(self, cols, stop_words='english'):
        self.stop_words = set()
        self.cols = None

        if isinstance(stop_words, str):
            if stop_words not in nltk_stopwords.fileids():
                raise ValueError("The language you entered is not available. Received: {}".format(stop_words))
            self.stop_words = nltk_stopwords.words(stop_words)
        elif isinstance(stop_words, list) and all(isinstance(elem, str) for elem in stop_words):
            for lang in stop_words:
                if lang not in nltk_stopwords.fileids():
                    raise ValueError("The following language you entered is not available: {}".format(lang))
                self.stop_words = self.stop_words.union(set(nltk_stopwords.words(lang)))
        else:
            raise TypeError("Please enter the language for your stop_words as a string or list of strings. Received: {}"
                            .format(type(stop_words)))

        if not cols:
            raise TypeError("You have to specify which columns you want to tokenize")
        if isinstance(cols, list) and all(isinstance(col, str) for col in cols):
            self.cols = cols
        else:
            raise TypeError("Cols has to be list of strings or a single string. Received: {}".format(type(cols)))

    def __eq__(self, other):
        """Overrides the default implementation"""
        return isinstance(other, self.__class__) and self.stop_words == other.stop_words and self.cols == other.cols

    def __hash__(self):
        """Overrides the default implementation"""
        hash_list = [self.__class__]
        hash_list.extend(sorted(self.stop_words))
        hash_list.extend(self.cols)
        return hash(tuple(hash_list))

    def make_usable_list(self, texts):
        """
        Make all words lowercase, remove stopwords, remove accents,
        remove words that are shorter than 2 chars or longer than 20 chars or that start with '_',
        convert words to unicode
        """
        tokenized = []
        for entry in texts:
            wordlist = []
            # Remove HTML tags
            entry = BeautifulSoup(entry, features="lxml").get_text(separator="")
            for word in re.sub(r"[^\w+\s]|\b[a-zA-Z]\b", ' ', entry).split():
                if word == '' or word.lower() in self.stop_words:
                    continue
                if len(gensim.utils.simple_preprocess(word)) > 0:
                    wordlist.append(gensim.utils.simple_preprocess(word, max_len=20, deacc=True).pop())
            tokenized.append(wordlist)
        return tokenized

    def process(self, store):
        df1_texts, df2_texts = store[ColumnType.text]

        for col in self.cols:
            if col not in store.column_names(ColumnType.text):
                raise ValueError("Given column is not contained in detected text columns of the datasets: {}"
                                 .format(col))
        col_names = self.cols

        processed1 = pd.DataFrame()
        processed2 = pd.DataFrame()

        for col in col_names:
            processed1[col] = self.make_usable_list(df1_texts[col])
            processed2[col] = self.make_usable_list(df2_texts[col])

        return processed1, processed2
