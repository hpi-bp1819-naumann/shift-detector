import pandas as pd
from typing import List, Dict
import logging as logger
from shift_detector.checks.Check import Check
from collections import defaultdict
from functools import reduce
from collections import namedtuple
from shift_detector.checks.Check import Reports
from pandas.api.types import is_numeric_dtype

CheckReports = namedtuple("CheckReports", "check reports")

class Detector:

    def __init__(self, first_path: str, second_path: str, separator=','):
        # TODO: remove sampling
        self.first_df = self.read_from_csv(first_path, separator).head(100)
        self.second_df = self.read_from_csv(second_path, separator).head(100)

        self.checks_to_run = []
        self.columns = []

        self.checks_reports = []

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

    @staticmethod
    def _is_categorical(col: pd.Series,
                        n_samples: int = 100,
                        max_unique_fraction=0.05) -> bool:
        """
        A heuristic to check whether a column is categorical:
        a column is considered categorical (as opposed to a plain text column)
        if the relative cardinality is max_unique_fraction or less.
        :param col: pandas Series containing strings
        :param n_samples: number of samples used for heuristic (default: 100)
        :param max_unique_fraction: maximum relative cardinality.
        :return: True if the column is categorical according to the heuristic
        """

        sample = col.sample(n=n_samples, replace=len(col) < n_samples).unique()

        return sample.shape[0] / n_samples < max_unique_fraction

    def _split_dataframes(self, df1: pd.DataFrame, df2: pd.DataFrame, columns: List[str]) -> Dict:
        """
        Split df1 and df2 in different dataframes related to type of the column.
        The column types are numeric, categorical and string.
        :param df1: first dataframe
        :param df2: second dataframe
        :param columns: the columns that both dataframes contain
        :return: dictionary that maps the column types to the splitted dataframes as tuples
        """
        numeric_columns = [c for c in columns if is_numeric_dtype(df1[c]) \
                            and is_numeric_dtype(df2[c])]
        categorical_columns = [c for c in columns if Detector._is_categorical(df1[c]) \
                                and Detector._is_categorical(df2[c])]
        string_columns = list(set(columns) - set(numeric_columns) - set(categorical_columns))

        return {
            "numeric": (df1[numeric_columns], df2[numeric_columns]),
            "categorical": (df1[categorical_columns], df2[categorical_columns]),
            "text": (df1[string_columns], df2[string_columns])
        }

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
            check_result = check.run()
            reports = Reports(check_result=check_result, report_class=check.report_class())
            check_reports = CheckReports(check=check, reports=reports)
            self.checks_reports.append(check_reports)

    ## Evaluate the results
    def evaluate(self):
        for check_report in self.checks_reports:
            check, reports = check_report
            print(check.name())
            for report in reports.reports:
                report.print_report()
