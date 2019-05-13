import pandas as pd
from typing import List
import logging as logger
from shift_detector.checks.Check import Check
from collections import defaultdict
from functools import reduce

class Detector:

    def __init__(self, first_path: str, second_path: str, separator=','):
        # TODO: remove sampling
        self.first_df = self.read_from_csv(first_path, separator).head(100)
        self.second_df = self.read_from_csv(second_path, separator).head(100)

        self.checks_to_run = []
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
    
    def add_check(self, check: Check):
        self.checks_to_run += [check]
        return self

    def add_checks(self, checks: List[Check]):
        self.checks_to_run += checks
        return self

    def get_result(self, index):
        """ Return the last calculated result for the check with the index """
        return self.checks_to_run[index].results[-1]

    def run(self):
        first_df_columns = list(self.first_df.head(0))
        second_df_columns = list(self.second_df.head(0))

        if first_df_columns != second_df_columns:
            logger.warning('The columns of the provided dataset '
                           'should be the same, but are {} and {}'.format(first_df_columns, second_df_columns))

            self.columns = self.get_common_column_names()
            logger.info('Using columns {} instead.'.format(self.columns))
        else:
            self.columns = first_df_columns

        if not self.checks_to_run:
            raise Exception('Please use the method add_test to '
                            'add tests that should be executed, before calling run()')

        if not self.checks_to_run:
            raise Exception('Please use the method add_test to \
                add tests that should be executed, before calling run()')

        ## Find column types
        # TODO: replace this provisional
        column_type_to_columns = {
            "int": (self.first_df[["marketplace_id", "refinement_id"]], self.second_df[["marketplace_id", "refinement_id"]]),
            "category": (self.first_df[["value", "attribute"]], self.second_df[["value", "attribute"]]),
            "text": (self.first_df["bullet_points"], self.second_df["bullet_points"])
        }

        def update_preprocessings(groups, checks):
            for key, value in checks.needed_preprocessing().items():
                groups[key].add(value)
            return groups

        type_to_needed_preprocessings = reduce(update_preprocessings, self.checks_to_run, defaultdict(set))
        type_to_needed_preprocessings = dict(type_to_needed_preprocessings)
        logger.info(f"Needed Preprocessing: {type_to_needed_preprocessings}")

        preprocessings = defaultdict(dict)
        '''
        preprocessings: {
            "int": {
                Default.Default: (pd.Dataframe1, pd.Dataframe2)
            }
        }
        '''
        ## Do the preprocessing
        for column_type, needed_preprocessings in type_to_needed_preprocessings.items():
            (first_df, second_df) = column_type_to_columns[column_type]
            for needed_preprocessing in needed_preprocessings:
                preprocessed = needed_preprocessing.process(first_df, second_df)
                preprocessings[column_type][needed_preprocessing] = preprocessed

        def choose_preprocessings(specific_preprocessings, pair):
            column_type, preprocessings_method = pair
            specific_preprocessings[column_type] = preprocessings[column_type][preprocessings_method]
            return specific_preprocessings

        ## Link the preprocessing and pass them to the checks
        for check in self.checks_to_run:
            chosen_preprocessing = reduce(choose_preprocessings, check.needed_preprocessing().items(), dict())
            check.set_data(chosen_preprocessing)

        ## Run the checks
        for check in self.checks_to_run:
            check.run()

    ## Evaluate the results
    def evaluate(self):
        for check in self.checks_to_run:
            print(check.name())
            for result in check.results:
                result.print_report()
