import pandas as pd
from typing import List
import logging as logger
from shift_detector.analyzers.analyzer import Analyzer


class Detector:

    def __init__(self, first_path: str, second_path: str, separator=','):
        # TODO: remove sample
        self.first_df = self.read_from_csv(first_path, separator).sample(100)
        self.second_df = self.read_from_csv(second_path, separator).sample(100)

        self.analyzers_to_run = []
        self.columns = []

    def read_from_csv(self, file_path: str, separator) -> pd.DataFrame:
        # TODO: give user feedback about how many lines were dropped
        logger.info('Reading in CSV file. This may take a while ...')
        return pd.read_csv(file_path, sep=separator, error_bad_lines=False).dropna()

    def get_common_column_names(self) -> List[str]:

        first_df_columns = list(self.first_df.head(0))
        second_df_columns = list(self.second_df.head(0))

        common_columns = set(first_df_columns).intersection(second_df_columns)

        if len(common_columns) == 0:
            raise Exception('The provided datasets do not have any column names in common. \
                They have {} and {}'.format(first_df_columns, second_df_columns))

        return list(common_columns)
    
    def add_analyzer(self, analyzer: Analyzer):
        
        self.analyzers_to_run += [analyzer]
        return self

    def add_analyzers(self, analyzers: List[Analyzer]):
        
        self.analyzers_to_run += analyzers
        return self

    def run(self):
        first_df_columns = list(self.first_df.head(0))
        second_df_columns = list(self.second_df.head(0))

        if (first_df_columns != second_df_columns):
            logger.error('The columns of the provided dataset \
                should be the same, but are {} and {}'.format(first_df_columns, second_df_columns))

            self.columns = self.get_common_column_names()
            logger.info('Using columns {} instead.'.format(self.columns))
        else:
            self.columns = first_df_columns

        if (self.analyzers_to_run == []):
            raise Exception('Please use the method add_test to \
                add tests that should be executed, before calling run()')

        for analyzer in self.analyzers_to_run:
            analyzer(self.first_df, self.second_df)\
                .run(self.columns)\
                .print_report()
