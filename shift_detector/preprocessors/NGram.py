from shift_detector.preprocessors.Preprocessor import Preprocessor
from enum import Enum
from shift_detector.Utils import ColumnType
from shift_detector.preprocessors.Store import Store

import pandas as pd

class NGramType(Enum):
    word = "word"
    character = "character"


class NGram(Preprocessor):

    def __init__(self, n: int, ngram_type: NGramType):
        self.n = n
        self.ngram_type = ngram_type
        if self.n < 1:
            raise ValueError('n has to be greater than 0')

    def __eq__(self, other):
        return self.n == other.n and self.ngram_type == other.ngram_type

    def __hash__(self):
        """
        Calculate hash for object based on the class type, n and the selected ngram_type
        :return: hash that represents the class with its settings
        """
        return hash((self.__class__, self.n, self.ngram_type))

    def generate_ngram(self, text: tuple) -> dict:
        ngram = {}
        for i in range(len(text) - self.n + 1):
            segment = tuple(text[i:i + self.n])
            ngram[segment] = 1 if segment not in ngram else ngram[segment] + 1
        return ngram

    def process(self, store: Store) -> tuple:
        # TODO: category should be text -> repair categorization
        train, test = store[ColumnType.categorical]
        train = train.copy()
        test = test.copy()
        for column in train:
            train[column] = train[column].dropna().str.lower()
            if self.ngram_type == NGramType.word:
                train[column] = train[column].str.split()
            train[column] = train[column].apply(lambda row: self.generate_ngram(tuple(row)))

        for column in test:
            test[column] = test[column].dropna().str.lower()
            if self.ngram_type == NGramType.word:
                test[column] = test[column].str.split()
            test[column] = test[column].apply(lambda row: self.generate_ngram(tuple(row)))

        return train, test
