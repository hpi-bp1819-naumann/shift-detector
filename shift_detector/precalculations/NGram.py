from shift_detector.precalculations.Preprocessor import Preprocessor
from enum import Enum
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

    def process(self, train: pd.Series, test: pd.Series) -> tuple:
        train = train.dropna().str.lower()
        test = test.dropna().str.lower()
        if self.ngram_type == NGramType.word:
            train = train.str.split()
            test = test.str.split()

        train_processed = train.apply(lambda row: self.generate_ngram(tuple(row)))
        test_processed = test.apply(lambda row: self.generate_ngram(tuple(row)))

        return train_processed, test_processed
