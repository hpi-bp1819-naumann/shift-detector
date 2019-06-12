import logging as logger

from pandas import DataFrame

from shift_detector.utils.ColumnManagement import split_dataframes, ColumnType, CATEGORICAL_MAX_RELATIVE_CARDINALITY
from shift_detector.utils.DataIO import shared_column_names

MIN_DATA_SIZE = int(CATEGORICAL_MAX_RELATIVE_CARDINALITY * 100)


class InsufficientDataError(Exception):

    def __init__(self, message, actual_size, expected_size):
        super().__init__(message)
        self.actual_size = actual_size
        self.expected_size = expected_size


class Store:

    def __init__(self,
                 df1: DataFrame,
                 df2: DataFrame):

        self.verify_min_data_size(min([len(df1), len(df2)]))

        self.columns = shared_column_names(df1, df2)
        self.df1 = df1[self.columns]
        self.df2 = df2[self.columns]

        self.splitted_dfs = split_dataframes(df1, df2, self.columns)
        self.preprocessings = dict()

    def __getitem__(self, needed_preprocessing) -> DataFrame:
        if isinstance(needed_preprocessing, ColumnType):
            return self.splitted_dfs[needed_preprocessing]

        """
        if not isinstance(needed_preprocessing, Preprocessor):
            raise Exception("Needed Preprocessing must be of type Preprocessor or ColumnType")
        """

        if needed_preprocessing in self.preprocessings:
            logger.info("Use already existing Preprocessing")
            return self.preprocessings[needed_preprocessing]

        logger.info("Execute new Preprocessing")
        preprocessing = needed_preprocessing.process(self)
        self.preprocessings[needed_preprocessing] = preprocessing
        return preprocessing

    @staticmethod
    def verify_min_data_size(size):
        if size < MIN_DATA_SIZE:
            raise InsufficientDataError('The input data is insufficient for the column type heuristics to work. Only '
                                        '{actual} row(s) were passed. Please pass at least {expected} rows.'
                                        .format(actual=size, expected=MIN_DATA_SIZE), size, MIN_DATA_SIZE)
