import pandas as pd
from typing import List, Dict
import logging as logger
from shift_detector.checks.Check import Check
from collections import defaultdict
from functools import reduce
from collections import namedtuple
from shift_detector.checks.Check import Reports
from pandas.api.types import is_numeric_dtype
from enum import Enum

class ColumnType(Enum):
    numeric = "numeric"
    categorical = "categorical"
    text = "text"

CheckReports = namedtuple("CheckReports", "check reports")

class Detector:

    def __init__(self, first_path: str, second_path: str, separator=','):
        # TODO: remove sampling
        self.first_df = self.read_from_csv(first_path, separator).sample(100)
        self.second_df = self.read_from_csv(second_path, separator).sample(100)

        self.checks_to_run = []
        self.checks_reports = []

    def add_check(self, check: Check):
        self.checks_to_run += [check]
        return self

    def add_checks(self, checks: List[Check]):
        self.checks_to_run += checks
        return self

    def read_from_csv(self, file_path: str, separator) -> pd.DataFrame:
        # TODO: give user feedback about how many lines were dropped
        logger.info('Reading in CSV file. This may take a while ...')
        return pd.read_csv(file_path, sep=separator, error_bad_lines=False).dropna()

    def _shared_column_names(self, df1: pd.DataFrame, df2: pd.DataFrame) -> List[str]:
        """
        Find the column names that both dataframes share.
        Raise an exception if the dataframes do not a shared column name.
        :param df1: first dataframe
        :param df2: second dataframe
        :return: List of the column names that both dataframes have. 
        """
        df1_columns = set(df1.columns.values)
        df2_columns = set(df2.columns.values)

        if df1_columns != df2_columns:
            logger.warning('The columns of the provided dataset \
                           should be the same, but are {} and {}'.format(df1_columns, df2_columns))

            shared_columns = df1_columns.intersection(df2_columns)

            if len(shared_columns) == 0:
                raise Exception('The provided datasets do not have any column names in common. \
                    They have {} and {}'.format(df1_columns, df2_columns))

            logger.info('Using columns {} instead.'.format(shared_columns))
        else:
            shared_columns = df1_columns

        return list(shared_columns)

    @staticmethod
    def _is_categorical(col: pd.Series,
                        n_samples: int = 100,
                        max_unique_fraction=0.1) -> bool:
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

        return sample.shape[0] / n_samples <= max_unique_fraction

    def _split_dataframes(self, df1: pd.DataFrame, df2: pd.DataFrame, columns: List[str]) -> Dict:
        """
        Split df1 and df2 in different dataframes related to type of the column.
        The column types are numeric, categorical and text.
        :param df1: first dataframe
        :param df2: second dataframe
        :param columns: the columns that both dataframes contain
        :return: dictionary that maps the column types to the splitted dataframes as tuples
        """
        numeric_columns = [c for c in columns if is_numeric_dtype(df1[c])
                            and is_numeric_dtype(df2[c])]
        logger.info("Assuming numerical columns: {}".format(", ".join(numeric_columns)))
        categorical_columns = [c for c in columns if Detector._is_categorical(df1[c])
                                and Detector._is_categorical(df2[c])]
        logger.info("Assuming categorical columns: {}".format(", ".join(categorical_columns)))
        text_columns = list(set(columns) - set(numeric_columns) - set(categorical_columns))
        logger.info("Assuming text columns: {}".format(", ".join(text_columns)))

        return {
            ColumnType.numeric: (df1[numeric_columns], df2[numeric_columns]),
            ColumnType.categorical: (df1[categorical_columns], df2[categorical_columns]),
            ColumnType.text: (df1[text_columns], df2[text_columns])
        }

    def _needed_preprocessing(self, checks: List[Check]) -> Dict:
        """
        Find the needed preprocessings for each column type and
        aggregate them.
        :param checks: the checks the prepocessings will be aggregated from
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
        type_to_needed_preprocessings = dict(type_to_needed_preprocessings)
        return type_to_needed_preprocessings

    def _preprocess(self,
                    column_type_to_columns: Dict,
                    type_to_needed_preprocessings: Dict) -> Dict:
        """
        Execute the preprocessing.
        :param column_type_to_columns: result of _split_dataframes
        :param type_to_needed_preprocessings: result of _needed_preprocessing
        :return: Dict
        {
            ColumnType: {
                Preprocessor: (df1_processed, df2_processed)
            }
        }
        """
        preprocessings = defaultdict(dict)

        for column_type, needed_preprocessings in type_to_needed_preprocessings.items():
            (first_df, second_df) = column_type_to_columns[column_type]
            for needed_preprocessing in needed_preprocessings:
                preprocessed = needed_preprocessing.process(first_df, second_df)
                preprocessings[column_type][needed_preprocessing] = preprocessed

        return preprocessings

    def run(self):
        columns = self._shared_column_names(self.first_df, self.second_df)

        if not self.checks_to_run:
            raise Exception('Please use the method add_test to \
                add tests that should be executed, before calling run()')

        ## Find column types
        column_type_to_columns = self._split_dataframes(self.first_df, self.second_df, columns)

        type_to_needed_preprocessings = self._needed_preprocessing(self.checks_to_run)
        logger.info(f"Needed Preprocessing: {type_to_needed_preprocessings}")

        ## Do the preprocessing
        preprocessings = self._preprocess(column_type_to_columns, type_to_needed_preprocessings)

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
