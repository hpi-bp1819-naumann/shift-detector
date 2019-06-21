import logging as logger
from collections import defaultdict
from typing import Union

import pandas as pd
from IPython.display import display, Markdown

from Morpheus.checks.check import Check, Report
from Morpheus.precalculations.store import Store
from Morpheus.utils.column_management import column_names
from Morpheus.utils.data_io import read_from_csv
from Morpheus.utils.neat_print import nprint


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
                 delimiter=',',
                 **custom_column_types):
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

        self.check_reports = []
        self.store = Store(self.df1, self.df2, custom_column_types)

        print("Used columns: {}".format(', '.join(column_names(self.store.column_names()))))

    def run(self, *checks, logger_level=logger.ERROR):
        """
        Run the Detector with the checks to run.
        :param checks: checks to run
        :param logger_level: level of logging
        """
        logger.getLogger().setLevel(logger_level)

        if not checks:
            raise Exception("Please include checks)")

        if not all(isinstance(check, Check) for check in checks):
            class_names = map(lambda c: c.__class__.__name__, checks)
            raise Exception("All elements in checks should be a Check. Received: {}".format(', '.join(class_names)))

        check_reports = []
        for check in checks:
            print("Executing {}".format(check.__class__.__name__))
            try:
                report = check.run(self.store)
                check_reports.append(report)
            except Exception as e:
                error_msg = {e.__class__.__name__: str(e)}
                error_report = Report(check.__class__.__name__,
                                      examined_columns=[],
                                      shifted_columns=[],
                                      information=error_msg)
                check_reports.append(error_report)
        self.check_reports = check_reports

    def evaluate(self):
        """
        Evaluate the reports.
        """
        nprint("OVERVIEW", text_formatting='h1')
        nprint("Executed {} checks".format(len(self.check_reports)))

        detected = defaultdict(int)
        examined = defaultdict(int)

        for report in self.check_reports:
            for shifted_column in report.shifted_columns:
                detected[shifted_column] += 1
            for examined_column in report.examined_columns:
                examined[examined_column] += 1

        def sort_key(t):
            """
            Sort descending with respect to number of failed checks.
            If two checks have the same number of failed, sort ascending
            with respect to the number of number of executed checks.
            """
            _, num_detected, num_examined = t

            return -num_detected, num_examined

        sorted_summary = sorted(((col, detected[col], examined[col]) for col in examined), key=sort_key)

        df_summary = pd.DataFrame(sorted_summary, columns=['Column', '# Shifts detected', '# Checks Executed'])
        display(df_summary)

        nprint("DETAILS", text_formatting='h1')
        for report in self.check_reports:
            report.print_report()
            for fig in report.figures:
                fig()
