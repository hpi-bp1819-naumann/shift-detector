import logging as logger
from typing import List, Union

import pandas as pd

from shift_detector.checks.Check import Check, Report
from shift_detector.precalculations.Store import Store
from shift_detector.utils.ColumnManagement import column_names
from shift_detector.utils.DataIO import read_from_csv


class Detector:
    """The detector object acts as the central object.
    It is passed the data frames you want to compare.

    :param df1: either a pandas data frame or a file path
    :param df2: either a pandas data frame or a file path
    :param delimiter: delimiter for csv files
    """

    def __init__(self,
                 df1: Union[pd.DataFrame, str],
                 df2: Union[pd.DataFrame, str],
                 delimiter=','):
        if type(df1) is pd.DataFrame:
            self.df1 = df1
        elif type(df1) is str:
            self.df1 = read_from_csv(df1, delimiter)
        else:
            raise Exception("df1 is not a dataframe or a string")

        if type(df2) is pd.DataFrame:
            self.df2 = df2
        elif type(df2) is str:
            self.df2 = read_from_csv(df2, delimiter)
        else:
            raise Exception("df2 is not a dataframe or a string")

        self.checks_to_run = []
        self.check_reports = []
        self.store = Store(self.df1, self.df2)

        logger.info("Used columns: {}".format(' '.join(column_names(self.store.columns))))

    def add_checks(self, checks):
        """
        Add checks to the detector

        :param checks: single or list of Checks
        """
        if isinstance(checks, Check):
            checks_to_run = [checks]
        elif isinstance(checks, list) and all(isinstance(check, Check) for check in checks):
            checks_to_run = checks
        else:
            raise Exception("All elements in checks should be a Check")
        self.checks_to_run += checks_to_run

    def run_checks(self) -> List[Report]:
        """
        Execute the checks to run.

        :return: list of Reports that resulted from the checks
        """
        return [check.run(self.store) for check in self.checks_to_run]

    def run(self):
        """
        Run the Detector.
        """
        if not self.checks_to_run:
            raise Exception("Please use the method add_checks to add checks, "
                            "that should be executed, before calling run()")

        self.check_reports = self.run_checks()

    def evaluate(self):
        """
        Evaluate the reports.
        """
        print("EVALUATION")
        for report in self.check_reports:
            print(report)
