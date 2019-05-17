from shift_detector.preprocessors.Preprocessor import Preprocessor


class Default(Preprocessor):

    def __eq__(self, other):
        """Overrides the default implementation"""
        return isinstance(other, self.__class__)

    def __hash__(self):
        """Overrides the default implementation"""
        return hash(self.__class__)

    def process(self, first_df, second_df):
        return first_df, second_df
