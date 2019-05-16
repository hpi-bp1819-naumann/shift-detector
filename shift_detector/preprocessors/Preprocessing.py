import logging as logger
from collections import defaultdict
from functools import reduce
from typing import List, Dict

import pandas as pd

from shift_detector.Utils import split_dataframes
from shift_detector.checks.Check import Check


def preprocess(checks: List[Check],
               df1: pd.DataFrame,
               df2: pd.DataFrame,
               columns: List[str]):
    """
    Preprocess the columns of df1 and df2 according to
    the preprocessings that the checks request.
    :param checks: checks to request the preprocessing from
    :param df1: first dataframe
    :param df2: second dataframe
    :param columns: the columns that will be processed
    :return: Dict
    {
        ColumnType: {
            Preprocessor: (df1_processed, df2_processed)
            ...
        }
        ...
    }
    """
    column_type_to_columns = split_dataframes(df1, df2, columns)
    logger.info("Splitted dataframes by column types")

    type_to_needed_preprocessings = Preprocessor.needed_preprocessing(checks)
    logger.info("Needed Preprocessing: {}".join(type_to_needed_preprocessings))

    preprocessings = Preprocessor.run_preprocessings(column_type_to_columns,
                                                     type_to_needed_preprocessings)
    logger.info("Executed Preprocessing")
    return preprocessings


class Preprocessor:

    @staticmethod
    def needed_preprocessing(checks: List[Check]) -> Dict:
        """
        Find the needed preprocessings for each column type and
        aggregate them.
        :param checks: the checks the preprocessings will be aggregated from
        :return: Dict
        {
            ColumnType: Set(Preprocessor, ...)
            ...
        }
        """

        def update_preprocessings(groups, checks):
            for key, value in checks.needed_preprocessing().items():
                groups[key].add(value)
            return groups

        type_to_needed_preprocessings = reduce(update_preprocessings, checks, defaultdict(set))
        return dict(type_to_needed_preprocessings)

    @staticmethod
    def run_preprocessings(column_type_to_columns: Dict,
                           type_to_needed_preprocessings: Dict) -> Dict:
        """
        Execute the preprocessing.
        :param column_type_to_columns: result of _split_dataframes
        :param type_to_needed_preprocessings: result of _needed_preprocessing
        :return: Dict
        {
            ColumnType: {
                Preprocessor: (df1_processed, df2_processed)
                ...
            }
            ...
        }
        """
        preprocessings = defaultdict(dict)

        for column_type, needed_preprocessings in type_to_needed_preprocessings.items():
            (first_df, second_df) = column_type_to_columns[column_type]
            for needed_preprocessing in needed_preprocessings:
                preprocessed = needed_preprocessing.process(first_df, second_df)
                preprocessings[column_type][needed_preprocessing] = preprocessed

        return dict(preprocessings)
