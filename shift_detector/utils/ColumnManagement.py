from enum import Enum
from typing import List, Dict
import logging as logger
import pandas as pd
from pandas.core.dtypes.common import is_numeric_dtype


class ColumnType(Enum):
    numerical = 'numerical'
    categorical = 'categorical'
    all_categorical = 'all_categorical'  # categorical and numeric-categorical
    text = 'text'


def is_categorical(col: pd.Series,
                   n_samples: int = 100,
                   max_unique_fraction: float = 0.1) -> bool:
    """
    A heuristic to check whether a column is categorical:
    a column is considered categorical (as opposed to a plain text column)
    if the relative cardinality is max_unique_fraction or less.
    :param col: pandas Series containing strings
    :param n_samples: number of samples used for heuristic (default: 100)
    :param max_unique_fraction: maximum relative cardinality.
    :return: True if the column is categorical according to the heuristic
    """

    n_samples = n_samples if len(col) >= n_samples else len(col)
    sample = col.sample(n=n_samples).unique()

    return sample.shape[0] / n_samples <= max_unique_fraction


def column_names(columns) -> List[str]:
    """
        Return the names of all input columns as list. If column is not named return index as string instead.
        :param columns: columns of a dataframe
        :return: List of column names
        """
    return [str(c) for c in columns]


def split_dataframes(df1: pd.DataFrame, df2: pd.DataFrame, columns: List) -> Dict:
    """
    Split df1 and df2 in different dataframes related to type of the column.
    The column types are numeric, categorical and text.
    :param df1: first dataframe
    :param df2: second dataframe
    :param columns: the columns that both dataframes contain
    :return: dictionary that maps the column types to the splitted dataframes as tuples
    {
        ColumnType: (df1_type, df2_type),
        ...
    }
    """
    numerical_columns = [c for c in columns if is_numeric_dtype(df1[c])
                       and is_numeric_dtype(df2[c])]
    logger.info("Assuming numerical columns: {}".format(", ".join(column_names(numerical_columns))))
    remaining_columns = list(set(columns) - set(numerical_columns))

    categorical_columns = [c for c in remaining_columns if is_categorical(df1[c]) and is_categorical(df2[c])]
    logger.info("Assuming categorical columns: {}".format(", ".join(column_names(categorical_columns))))

    numeric_categorical_columns = [c for c in numerical_columns if is_categorical(df1[c]) and is_categorical(df2[c])]

    all_categorical_columns = categorical_columns.copy()
    all_categorical_columns.extend(numeric_categorical_columns)

    text_columns = list(set(columns) - set(numerical_columns) - set(categorical_columns))
    logger.info("Assuming text columns: {}".format(", ".join(column_names(text_columns))))

    return {
        ColumnType.numerical: (df1[numerical_columns], df2[numerical_columns]),
        ColumnType.categorical: (df1[categorical_columns], df2[categorical_columns]),
        ColumnType.all_categorical: (df1[all_categorical_columns], df2[all_categorical_columns]),
        ColumnType.text: (df1[text_columns], df2[text_columns])
    }
