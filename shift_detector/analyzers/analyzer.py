import pandas as pd
from abc import ABCMeta, abstractmethod
from datawig.utils import logger as datawig_logger

class AnalyzerResult():

    __metaclass__ = ABCMeta

    def __init__(self, result={}):
        self.result = result


    @abstractmethod
    def print_report(self):
        """

        Print report for analyzed columns

        """
        pass



class Analyzer():

    __metaclass__ = ABCMeta

    def __init__(self, data1: pd.DataFrame, data2: pd.DataFrame):

        if data1 is None:
            raise Exception('No dataframe provided for argument first_df')

        if data2 is None:
            raise Exception('No dataframe provided for argument second_df')

        self.data1 = data1
        self.data2 = data2

        datawig_logger.setLevel('ERROR')


    @abstractmethod
    def run(self, columns=[]) -> AnalyzerResult:
        """

        Runs analyzer on provided columns

        :param columns:
        :return: AnalyzerResult

        """
        pass
