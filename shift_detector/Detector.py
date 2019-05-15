import logging as logger
from collections import namedtuple
from functools import reduce
from typing import List, Dict, Union

import pandas as pd

from shift_detector.Utils import shared_column_names, read_from_csv
from shift_detector.checks.Check import Check
from shift_detector.checks.Check import Reports
from shift_detector.preprocessors.Preprocessor import preprocess

CheckReports = namedtuple("CheckReports", "check reports")


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
        self.checks_reports = []
        self.column_type_to_columns = {}
        self.preprocessings = {}

    def add_check(self, check: Check):
        self.checks_to_run += [check]
        return self

    def add_checks(self, checks: List[Check]):
        self.checks_to_run += checks
        return self

    def _distribute_preprocessings(self, checks: List[Check], preprocessings: Dict):
        """
        Distribute the preprocessings to the checks.
        :param checks: checks to distribute the preprocessing to
        :param preprocessings: result of _preprocess
        """

        def choose_preprocessings(specific_preprocessings, pair):
            column_type, preprocessings_method = pair
            specific_preprocessings[column_type] = preprocessings[column_type][preprocessings_method]
            return specific_preprocessings

        for check in checks:
            chosen_preprocessing = reduce(choose_preprocessings, check.needed_preprocessing().items(), dict())
            check.set_data(chosen_preprocessing)

    def _run_checks(self, checks: List[Check]) -> List[CheckReports]:
        """
        Execute the checks.
        :param checks: the checks that will be executed
        :return: list of CheckReports, that connects runned checks with their reports
        """
        checks_reports = []
        for check in checks:
            check_result = check.run()
            reports = Reports(check_result=check_result, report_class=check.report_class())
            check_reports = CheckReports(check=check, reports=reports)
            checks_reports.append(check_reports)

        return checks_reports

    def run(self):
        columns = shared_column_names(self.first_df, self.second_df)
        logger.info(f"Used columns: {columns}")

        if not self.checks_to_run:
            raise Exception('Please use the method add_check to add checks, '
                            'that should be executed, before calling run()')

        preprocessings = preprocess(self.checks_to_run, self.first_df, self.second_df, columns)
        logger.info("Finished Preprocessing")

        self._distribute_preprocessings(self.checks_to_run, preprocessings)
        logger.info("Distributed Preprocessings to the Checks")

        self.checks_reports = self._run_checks(self.checks_to_run)

    ## Evaluate the results
    def evaluate(self):
        print("EVALUATION")
        for check_report in self.checks_reports:
            check, reports = check_report
            print(check.name())
            for report in reports.reports:
                report.print_report()
