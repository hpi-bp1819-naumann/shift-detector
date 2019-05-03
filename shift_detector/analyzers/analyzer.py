import pandas as pd
from abc import ABCMeta, abstractmethod
from datawig.utils import logger as datawig_logger

class AnalyzerResult(metaclass=ABCMeta):

    def __init__(self, result={}):
        self.result = result

    @abstractmethod
    def print_report(self):
        """

        Print report for analyzed columns

        """
        pass


class Analyzer(metaclass=ABCMeta):

    def __init__(self, first_df: pd.DataFrame, second_df: pd.DataFrame):

        if first_df is None:
            raise Exception('No dataframe provided for argument first_df')

        if second_df is None:
            raise Exception('No dataframe provided for argument second_df')

        self.first_df = first_df
        self.second_df = second_df

        datawig_logger.setLevel('ERROR')

    @abstractmethod
    def run(self, columns=[]) -> AnalyzerResult:
        """

        Runs analyzer on provided columns

        :param columns:
        :return: AnalyzerResult

        """
        pass
