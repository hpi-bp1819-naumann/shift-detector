import re
import pandas as pd

from shift_detector.precalculations.precalculation import Precalculation
from shift_detector.utils.column_management import ColumnType


class TokenizeIntoLowerWordsPrecalculation(Precalculation):

    def __eq__(self, other):
        return isinstance(other, self.__class__)

    def __hash__(self):
        return hash(self.__class__)

    @staticmethod
    def tokenize_into_words(text):
        text = text.lower()
        text = re.sub(r"-", ' ', text)
        text = re.sub(r"[^\w\s']", '', text)
        text = re.sub(r"\s+", ' ', text)
        if text == '' or text == ' ':
            return []
        if text[0] == ' ':
            text = text[1:]
        if text[-1] == ' ':
            text = text[:-1]
        splitted = re.split(r'\W\s|\s', text)
        return splitted

    def process(self, store):
        tokenized1 = pd.DataFrame()
        tokenized2 = pd.DataFrame()
        df1, df2 = store[ColumnType.text]
        for column in df1.columns:
            clean1 = df1[column].dropna()
            clean2 = df2[column].dropna()
            tokenized1[column] = [self.tokenize_into_words(text) for text in clean1]
            tokenized2[column] = [self.tokenize_into_words(text) for text in clean2]
        return tokenized1, tokenized2
