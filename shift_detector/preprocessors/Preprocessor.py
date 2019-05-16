from abc import ABCMeta, abstractmethod


class Preprocessor(metaclass=ABCMeta):

    @abstractmethod
    def __eq__(self, other):
        """Overrides the default implementation"""
        pass

    @abstractmethod
    def __hash__(self):
        """Overrides the default implementation"""
        pass

    @abstractmethod
    def process(self, first_df, second_df):
        pass
