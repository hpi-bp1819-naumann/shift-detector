import logging as logger
from typing import List

import pandas as pd


def read_from_csv(file_path: str, separator: str) -> pd.DataFrame:
    logger.info('Reading in CSV file. This may take a while ...')
    return pd.read_csv(file_path, sep=separator, error_bad_lines=False)


def shared_column_names(df1: pd.DataFrame, df2: pd.DataFrame) -> List[str]:
    """
    Find the column names that both dataframes share.
    Raise an exception if the dataframes do not have a shared column name.
    :param df1: first dataframe
    :param df2: second dataframe
    :return: List of the column names that both dataframes have.
    """
    df1_columns = df1.columns.values
    df2_columns = df2.columns.values

    if set(df1_columns) != set(df2_columns):
        logger.warning("The columns of the provided dataset should be the same, "
                       "but are {} for df1 and {} for df2".format(df1_columns, df2_columns))

        shared_columns = [column for column in df1_columns if column in df2_columns]

        if len(shared_columns) == 0:
            raise Exception("The provided datasets do not have any column names in common.")
    else:
        shared_columns = df1_columns

    return list(shared_columns)
