from pandas import DataFrame

from shift_detector.Utils import ColumnType
from shift_detector.preprocessors.Preprocessor import Preprocessor
from shift_detector.preprocessors.Store import Store


class DummyPreprocessing(Preprocessor):

    def __init__(self, dummy_parameter=2):
        self.parameter = dummy_parameter

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.parameter == other.parameter

    def __hash__(self):
        return hash(tuple([self.__class__, self.parameter]))

    def process(self, store: Store) -> (DataFrame, DataFrame):
        df1_numerical, df2_numerical = store[ColumnType.numeric]
        df1_processed = df1_numerical + self.parameter
        df2_processed = df2_numerical + self.parameter
        return df1_processed, df2_processed
