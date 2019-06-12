import pandas as pd
from pandas import DataFrame

from shift_detector.precalculations.precalculation import Precalculation
from shift_detector.utils.column_management import ColumnType, is_categorical


class BinningPrecalculation(Precalculation):

    def __init__(self, bins=50):
        self.bins = bins

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.bins == other.bins

    def __hash__(self):
        return hash(tuple([self.__class__, self.bins]))

    def process(self, store):
        """
        Bin the numerical columns of the datasets that do not
        suffice the categorical criterion.
        :param store: Store
        :return: The binned version of numerical columns
        """
        df1_numerical, df2_numerical = store[ColumnType.numerical]

        df1_size = len(df1_numerical)
        df2_size = len(df2_numerical)

        dfs = pd.concat([df1_numerical, df2_numerical])
        dfs_binned = DataFrame()

        for column_name in list(df1_numerical.columns):
            column = dfs[column_name]
            if is_categorical(column):
                dfs_binned[column_name] = column
            else:
                column_name_binned = "{}_binned".format(column_name)
                dfs_binned[column_name_binned] = pd.cut(column, self.bins)

        df1_binned = dfs_binned.head(df1_size)
        df2_binned = dfs_binned.tail(df2_size)

        return df1_binned, df2_binned
