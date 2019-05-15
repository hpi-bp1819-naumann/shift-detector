from shift_detector.preprocessors.Preprocessor import Preprocessor
from abc import ABCMeta, abstractmethod


class NGram(Preprocessor, metaclass=ABCMeta):

    def __init__(self, n):
        self.n = n
        if self.n < 1:
            raise ValueError('n has to be greater than 0')

    def __eq__(self, other):
        """Overrides the default implementation"""
        return self.n == other.n

    def __hash__(self):
        return hash((self.__class__, self.n))

    @abstractmethod
    def process(self, first_df, second_df):
        return first_df, second_df