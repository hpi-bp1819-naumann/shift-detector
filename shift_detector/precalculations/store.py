import logging as logger

from pandas import DataFrame

from shift_detector.utils.column_management import detect_column_types, ColumnType, CATEGORICAL_MAX_RELATIVE_CARDINALITY
from shift_detector.utils.data_io import shared_column_names

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

        self.shared_columns = shared_column_names(df1, df2)
        self.df1 = df1[self.shared_columns]
        self.df2 = df2[self.shared_columns]

        self.type_to_columns = detect_column_types(self.df1, self.df2, self.shared_columns)
        self.splitted_dfs = {column_type: (self.df1[columns], self.df2[columns])
                             for column_type, columns in self.type_to_columns.items()}
        self.preprocessings = {}

    def __getitem__(self, needed_preprocessing) -> DataFrame:
        if isinstance(needed_preprocessing, ColumnType):
            return self.splitted_dfs[needed_preprocessing]
        '''
        if not isinstance(needed_preprocessing, Precalculation):
            raise Exception("Needed Preprocessing must be of type Precalculation or ColumnType")
        '''
        if needed_preprocessing in self.preprocessings:
            logger.info("Use already existing Precalculation")
            return self.preprocessings[needed_preprocessing]

        logger.info("Execute new Precalculation")
        preprocessing = needed_preprocessing.process(self)
        self.preprocessings[needed_preprocessing] = preprocessing
        return preprocessing

    def column_names(self, *column_types):
        if not column_types:
            return self.shared_columns

        if any([not isinstance(column_type, ColumnType) for column_type in column_types]):
            raise TypeError("column_types should be empty or of type ColumnType.")

        multi_columns = [self.type_to_columns[column_type] for column_type in column_types]
        flattened = {column for columns in multi_columns for column in columns}
        return list(flattened)

    @staticmethod
    def verify_min_data_size(size):
        if size < MIN_DATA_SIZE:
            raise InsufficientDataError('The input data is insufficient for the column type heuristics to work. Only '
                                        '{actual} row(s) were passed. Please pass at least {expected} rows.'
                                        .format(actual=size, expected=MIN_DATA_SIZE), size, MIN_DATA_SIZE)
