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
        
    @abstractmethod
    def run(self, columns=[]) -> CheckResult:
        """

        Runs check on provided columns

        :param columns:
        :return: CheckResult

        """
        pass
