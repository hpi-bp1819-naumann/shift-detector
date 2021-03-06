import logging as logger
from collections import defaultdict
from typing import Union

import pandas as pd
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors


from shift_detector.checks.check import Check, Report
from shift_detector.precalculations.store import Store
from shift_detector.utils.column_management import column_names
from shift_detector.utils.custom_print import nprint, lprint
from shift_detector.utils.data_io import read_from_csv


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
                 log_print=True,
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

        self.log_print = log_print
        self.check_reports = []
        self.store = Store(self.df1, self.df2, log_print=self.log_print, custom_column_types=custom_column_types)

        lprint("Used columns: {}".format(', '.join(column_names(self.store.column_names()))), self.log_print)

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
            lprint("Executing {}".format(check.__class__.__name__), self.log_print)
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
        nprint("Executed {} check{}".format(len(self.check_reports), 's' if len(self.check_reports) > 1 else ''))

        detected = defaultdict(int)
        examined = defaultdict(int)
        check_names = {}

        for report in self.check_reports:
            check_names[report.check_name] = []
            for shifted_column in report.shifted_columns:
                detected[shifted_column] += 1
                check_names[report.check_name].append(shifted_column)
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

        #sorted_summary = sorted(((col, detected[col], examined[col]) for col in examined), key=sort_key)

        #df_summary = pd.DataFrame(sorted_summary, columns=['Column', '# Shifts detected', '# Checks Executed'])

        check_matrix = [None]*len(check_names.keys())
        for i, check in enumerate(check_names.keys()):
            check_list = [None]*len(examined)
            for j, col in enumerate(examined):
                if col in check_names[check]:
                    check_list[j] = 1
                else:
                    check_list[j] = 0
            check_matrix[i] = check_list

        custom_cmap = colors.ListedColormap(['green', 'red'])

        fig, ax = plt.subplots()
        im = ax.imshow(check_matrix, cmap=custom_cmap, interpolation='none', vmin=0, vmax=1)

        # We want to show all ticks...
        ax.set_xticks(np.arange(len(examined)))
        ax.set_yticks(np.arange(len(check_names.keys())))
        # ... and label them with the respective list entries
        ax.set_xticklabels(examined)
        ax.set_yticklabels(check_names.keys())

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Minor ticks
        ax.set_xticks(np.arange(-.5, len(examined), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(check_names.keys()), 1), minor=True)
        ax.grid(which='minor', color='k', linestyle='-', linewidth=2)

        display(plt.show())

        nprint("DETAILS", text_formatting='h1')
        for report in self.check_reports:
            report.print_report()
            for fig in report.figures:
                fig()
