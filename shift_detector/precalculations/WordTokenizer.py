import gensim
from shift_detector.precalculations.Precalculation import Precalculation
from shift_detector.Utils import ColumnType
from nltk.corpus import stopwords as nltk_stopwords
import re


class WordTokenizer(Precalculation):

    def __init__(self, stop_words='english'):
        self.stopwords = None
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
            raise Exception('Please enter the language for your stopwords as a string or list of strings')
    
    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, self.__class__) and self.stop_words == other.stop_words:
            return True
        return False

    def __hash__(self):
        """Overrides the default implementation"""
        return hash(tuple([self.__class__] + sorted(self.stop_words)))

    def process(self, store):
        df1_texts, df2_texts = store[ColumnType.text]
        col_names = df1_texts.columns
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

        return processed1, processed2
