from abc import ABCMeta


class Preprocessor(metaclass=ABCMeta):

    @staticmethod
    def __eq__(self, other):
        """Overrides the default implementation"""
        return hash(self) == hash(other)

    @staticmethod
    def __hash__(self):
        """Overrides the default implementation"""
        return hash(self.__class__)

    @staticmethod
    def process(self, first_df, second_df):
        return first_df, second_df

    @staticmethod
    def static_process(first_df, second_df):
        return first_df, second_df
