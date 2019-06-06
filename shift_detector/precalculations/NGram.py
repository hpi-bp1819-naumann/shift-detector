from shift_detector.precalculations.Preprocessor import Preprocessor
from enum import Enum
from shift_detector.precalculations.Store import Store
from shift_detector.utils.ColumnManagement import ColumnType


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
        """
        Generates an ngram from a given text, n is defined by the class
        :param text: tuple that contains the atomic elements of the text (e.g. characters or words)
        :return: a dictionary representing the ngram
        """
        ngram = {}
        for i in range(len(text) - self.n + 1):
            segment = tuple(text[i:i + self.n])
            ngram[segment] = 1 if segment not in ngram else ngram[segment] + 1
        return ngram

    def process(self, store: Store) -> tuple:
        """
        :param store:
        :return: a tuple that contains the processed dataframes
        """

        df1, df2 = store[ColumnType.text]
        df1 = df1.copy()
        df2 = df2.copy()

        for column in df1:
            df1[column] = df1[column].dropna().str.lower()
            if self.ngram_type == NGramType.word:
                df1[column] = df1[column].str.split()
            df1[column] = df1[column].apply(lambda row: self.generate_ngram(tuple(row)))

        for column in df2:
            df2[column] = df2[column].dropna().str.lower()
            if self.ngram_type == NGramType.word:
                df2[column] = df2[column].str.split()
            df2[column] = df2[column].apply(lambda row: self.generate_ngram(tuple(row)))

        return df1, df2
