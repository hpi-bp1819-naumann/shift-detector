import logging as logger
from typing import List

from pandas import DataFrame

from shift_detector.Utils import split_dataframes, ColumnType
from shift_detector.preprocessors.Preprocessor import Preprocessor


class Store:

    def __init__(self,
                 df1: DataFrame,
                 df2: DataFrame,
                 shared_columns: List[str]):

        self.df1 = df1
        self.df2 = df2
        self.columns = shared_columns
        self.splitted_dfs = split_dataframes(df1, df2, shared_columns)

        self.preprocessings = dict()

    def __getitem__(self, needed_preprocessing: Preprocessor) -> DataFrame:
        if needed_preprocessing in self.preprocessings:
            logger.info("Use already existing Preprocessing")
            return self.preprocessings[needed_preprocessing]

        logger.info("Execute new Preprocessing")
        preprocessing = needed_preprocessing.process(self)
        self.preprocessings[needed_preprocessing] = preprocessing
        return preprocessing

    def numerical(self) -> (DataFrame, DataFrame):
        return self.splitted_dfs[ColumnType.numeric]
