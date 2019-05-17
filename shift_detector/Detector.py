import logging as logger
from typing import List, Union

import pandas as pd

from shift_detector.Utils import shared_column_names, read_from_csv
from shift_detector.checks.Check import Check, Report
from shift_detector.preprocessors.Store import Store


class Detector:

    def __init__(self,
                 df1: Union[pd.DataFrame, str],
                 df2: Union[pd.DataFrame, str],
                 delimiter=','):
        """
        :param df1: either a dataframe or the file path
        :param df2: either a dataframe or the file path
        :param delimiter: used delimiter for csv files
        """
        # TODO: remove sampling
        if type(df1) is pd.DataFrame:
            self.first_df = df1
        elif type(df1) is str:
            self.first_df = read_from_csv(df1, delimiter).sample(100)
        else:
            raise Exception("df1 is not a dataframe or a string")

        if type(df2) is pd.DataFrame:
            self.second_df = df2
        elif type(df2) is str:
            self.second_df = read_from_csv(df1, delimiter).sample(100)
        else:
            raise Exception("df2 is not a dataframe or a string")

        self.checks_to_run = []
        self.check_reports = []
        self.store = Store(self.first_df, self.second_df)

    def add_check(self, check: Check):
        self.checks_to_run += [check]
        return self

    def add_checks(self, checks: List[Check]):
        self.checks_to_run += checks
        return self

    def run_checks(self) -> List[Report]:
        """
        Execute the checks to run.
        :return: list of Reports that resulted from the checks
        """
        return [check.run(self.store) for check in self.checks_to_run]

    def run(self):
        if not self.checks_to_run:
            raise Exception('Please use the method add_check to add checks, '
                            'that should be executed, before calling run()')

        self.check_reports = self.run_checks()

    # Evaluate the results
    def evaluate(self):
        print("EVALUATION")
        for report in self.check_reports:
            report.print_report()
