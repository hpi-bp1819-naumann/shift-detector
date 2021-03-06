from enum import Enum
from typing import List

import pandas as pd
from pandas.core.dtypes.common import is_numeric_dtype

CATEGORICAL_MAX_RELATIVE_CARDINALITY = 0.1  # maximum ratio of distinct values in a categorical column


class ColumnType(Enum):
    numerical = 'numerical'
    categorical = 'categorical'
    text = 'text'


def is_categorical(col: pd.Series,
                   n_samples: int = 100,
                   max_unique_fraction: float = CATEGORICAL_MAX_RELATIVE_CARDINALITY) -> bool:
    """
    A heuristic to check whether a column is categorical:
    a column is considered categorical (as opposed to a plain text column)
    if the relative cardinality is max_unique_fraction or less.
    :param col: pandas Series containing strings
    :param n_samples: maximum sample size used for heuristic (default: 100) if series is shorter all values are used
    :param max_unique_fraction: maximum relative cardinality.
    :return: True if the column is categorical according to the heuristic
    """

    if len(col) >= n_samples:
        sample = col.sample(n=n_samples)
        unique_fraction = sample.unique().shape[0] / n_samples
    else:
        sample = col
        unique_fraction = sample.unique().shape[0] / sample.shape[0]

    return unique_fraction <= max_unique_fraction


def is_binary(col: pd.Series, allow_na=True):
    if allow_na:
        col = col.dropna()
    values = sorted(col.unique())
    return values == [0, 1]


def column_names(columns) -> List[str]:
    """
        Return the names of all input columns as list. If column is not named return index as string instead.
        :param columns: columns of a dataframe
        :return: List of column names
        """
    return [str(c) for c in columns]


def detect_column_types(df1, df2, columns):
    """
    Split df1 and df2 in different dataframes related to type of the column.
    The column types are numeric, categorical and text.
    :param df1: first dataframe
    :param df2: second dataframe
    :param columns: the columns that both dataframes contain
    :return: dictionary that maps the column types to the respective columns
    {
        ColumnType: [column1, ...],
        ...
    }
    """
    categorical_columns = [c for c in columns if is_binary(df1[c]) and is_binary(df2[c])]

    remaining_columns = list(set(columns) - set(categorical_columns))
    numerical_columns = [c for c in remaining_columns if is_numeric_dtype(df1[c]) and is_numeric_dtype(df2[c])]

    non_numerical = list(set(remaining_columns) - set(numerical_columns))
    categorical_columns.extend([c for c in non_numerical if is_categorical(df1[c]) and is_categorical(df2[c])])

    text_columns = list(set(non_numerical) - set(categorical_columns))

    return {
        ColumnType.numerical: numerical_columns,
        ColumnType.categorical: categorical_columns,
        ColumnType.text: text_columns
    }

