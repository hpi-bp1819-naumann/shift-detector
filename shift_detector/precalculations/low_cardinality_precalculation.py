import pandas as pd

from shift_detector.precalculations.precalculation import Precalculation
from shift_detector.utils.column_management import ColumnType
from shift_detector.utils.column_management import is_categorical


class LowCardinalityPrecalculation(Precalculation):

    def __eq__(self, other):
        return isinstance(other, LowCardinalityPrecalculation)

    def __hash__(self):
        return hash(LowCardinalityPrecalculation)

    def process(self, store):
        df1_numerical, df2_numerical = store[ColumnType.numerical]
        numerical_columns = store.column_names(ColumnType.numerical)

        low_cardinal_numerical_columns = [c for c in numerical_columns if
                                          is_categorical(df1_numerical[c]) and is_categorical(df2_numerical[c])]

        categorical_columns = store.column_names(ColumnType.categorical)

        low_cardinal_columns = categorical_columns.copy()
        low_cardinal_columns.extend(low_cardinal_numerical_columns)

        df1_categorical, df2_categorical = store[ColumnType.categorical]

        df1 = pd.concat([df1_categorical, df1_numerical[low_cardinal_numerical_columns]], axis=1)
        df2 = pd.concat([df2_categorical, df2_numerical[low_cardinal_numerical_columns]], axis=1)

        return df1, df2, low_cardinal_columns
