import pandas as pd
from abc import ABCMeta, abstractmethod
from datawig.utils import logger as datawig_logger


class CheckResult(metaclass=ABCMeta):

    def __init__(self, result={}):
        self.result = result

    @abstractmethod
    def print_report(self):
        """

        Print report for checked columns

        """


class Check(metaclass=ABCMeta):

    def __init__(self, df1: pd.DataFrame, df2: pd.DataFrame):

        if df1 is None:
            raise Exception('No dataframe provided for argument first_df')

        if df2 is None:
            raise Exception('No dataframe provided for argument second_df')

        self.df1 = df1
        self.df2 = df2

        datawig_logger.setLevel('ERROR')
        
    @abstractmethod
    def run(self, columns=[]) -> CheckResult:
        """

        Runs check on provided columns

        :param columns:
        :return: CheckResult

        """
        pass
