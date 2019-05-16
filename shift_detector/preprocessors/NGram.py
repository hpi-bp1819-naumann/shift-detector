from shift_detector.preprocessors.Preprocessor import Preprocessor
from abc import ABCMeta, abstractmethod
from enum import Enum


class NGramType(Enum):
    word = "word"
    character = "character"


class NGram(Preprocessor):

    def __init__(self, n: int, ngram_type : NGramType):
        self.n = n
        self.ngram_type = ngram_type
        if self.n < 1:
            raise ValueError('n has to be greater than 0')

    def __eq__(self, other):
        """Overrides the default implementation"""
        return self.n == other.n and self.ngram_type == other.ngram_type

    def __hash__(self):
        return hash((self.__class__, self.n, self.ngram_type))

    def generate_ngram(self, text):
        ngram = {}
        for i in range(len(text) - self.n + 1):
            ngram[tuple(text[i:i + self.n])] = 1 if tuple(text[i:i + self.n]) not in ngram \
                else ngram[tuple(text[i:i + self.n])] + 1
        return ngram

    def process(self, train, test):
        if self.ngram_type == NGramType.word:
            train = train.dropna().str.lower().str.split()
            test = test.dropna().str.lower().str.split()
        else:
            train = train.dropna().str.lower()
            test = test.dropna().str.lower()

        train_processed = train.apply(lambda row: self.generate_ngram(list(row)))
        test_processed = test.apply(lambda row: self.generate_ngram(list(row)))

        return train_processed, test_processed
