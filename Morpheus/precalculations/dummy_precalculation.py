from pandas import DataFrame

from Morpheus.precalculations.precalculation import Precalculation
from Morpheus.precalculations.store import Store
from Morpheus.utils.column_management import ColumnType


class DummyPrecalculation(Precalculation):

    def __init__(self, dummy_parameter=2):
        self.parameter = dummy_parameter

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.parameter == other.parameter

    def __hash__(self):
        return hash(tuple([self.__class__, self.parameter]))

    def process(self, store: Store) -> (DataFrame, DataFrame):
        df1_numerical, df2_numerical = store[ColumnType.numerical]
        df1_processed = df1_numerical + self.parameter
        df2_processed = df2_numerical + self.parameter
        return df1_processed, df2_processed
