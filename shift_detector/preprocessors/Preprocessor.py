from abc import ABCMeta, abstractmethod


class Preprocessor(metaclass=ABCMeta):

    @abstractmethod
    def __eq__(self, other):
        """Overrides the default implementation"""
        return hash(self) == hash(other)

    @abstractmethod
    def __hash__(self):
        """Overrides the default implementation"""
        return hash(self.__class__)

    @abstractmethod
    def process(self, first_df, second_df):
        return first_df, second_df
