import logging as logger

from pandas import DataFrame

from shift_detector.Utils import split_dataframes, ColumnType, shared_column_names, CATEGORICAL_MAX_RELATIVE_CARDINALITY

MIN_DATA_SIZE = CATEGORICAL_MAX_RELATIVE_CARDINALITY * 100


class InsufficientDataError(Exception):

    def __init__(self, message, min_size):
        super().__init__(message)
        self.min_size = min_size


class Store:

    def __init__(self,
                 df1: DataFrame,
                 df2: DataFrame):

        if len(df1) < MIN_DATA_SIZE or len(df2) < MIN_DATA_SIZE:
            raise InsufficientDataError('The input data is insufficient for the column type heuristics to work.',
                                        MIN_DATA_SIZE)

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
