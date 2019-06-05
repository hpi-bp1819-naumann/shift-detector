import gensim
from shift_detector.precalculations.Precalculation import Precalculation
from shift_detector.Utils import ColumnType
from nltk.corpus import stopwords as nltk_stopwords


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
            raise Exception('Please enter the language for your stopwords as a string or list of strings')
    
    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, self.__class__):
            if self.stop_words == other.stop_words:
                return True
        return False

    def __hash__(self):
        """Overrides the default implementation"""
        return hash(tuple([self.__class__] + sorted(self.stop_words)))

    def process(self, store):
        train_texts, test_texts = store[ColumnType.text]
        # TODO: do everything in the process function
        processed1 = [[[word for word in gensim.utils.simple_preprocess(str(doc), deacc=True)
                       if word not in self.stopwords] for doc in texts] for texts in train_texts]
        processed2 = [[[word for word in gensim.utils.simple_preprocess(str(doc), deacc=True)
                        if word not in self.stopwords] for doc in texts] for texts in test_texts]

        return processed1, processed2
