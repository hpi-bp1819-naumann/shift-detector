import logging as logger
from collections import defaultdict
from typing import List, Union

import pandas as pd

from shift_detector.Utils import read_from_csv, column_names
from shift_detector.checks.Check import Check, Report
from shift_detector.precalculations.Store import Store


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

        logger.info("Used columns: {}".format(', '.join(column_names(self.store.columns))))

    @staticmethod
    def detect(df1, df2, *checks, delimiter=','):
        detector = Detector(df1, df2, delimiter)
        detector.add_checks(checks)
        detector.run()
        detector.evaluate()
        return detector

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

    def run(self, *checks):
        """
        Run the Detector with the checks to run.
        :param checks: checks to run
        """
        if not checks:
            raise Exception("Please use the method add_checks to add checks, "
                            "that should be executed, before calling run()")

        self.check_reports = [check.run(self.store) for check in checks]

    def evaluate(self):
        """
        Evaluate the reports.
        """
        print("OVERVIEW")
        detected = defaultdict(int)
        examined = defaultdict(int)

        for report in self.check_reports:
            for shifted_column in report.shifted_columns:
                detected[shifted_column] += 1
            for examined_column in report.examined_columns:
                examined[examined_column] += 1

        sorted_columns = sorted(((col, detected[col], examined[col]) for col in examined), key=lambda t: (-t[1], t[2]))

        df = pd.DataFrame(sorted_columns, columns=['Column', '# Checks Failed', '# Checks Executed'])
        print(df, '\n')

        print("DETAILS")
        for report in self.check_reports:
            print(report)
            '''
            for fig in report.figures:
                fig()
            '''
