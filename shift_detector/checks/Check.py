import pandas as pd
from abc import ABCMeta, abstractmethod
from datawig.utils import logger as datawig_logger

class Check(metaclass=ABCMeta):

    def __init__(self):
        self.data = dict()

    @staticmethod
    @abstractmethod
    def name() -> str:
        """

        :return: Name of the Check

        """
        pass

    @staticmethod
    @abstractmethod
    def report_class():
        """

        :return: The class that will be used for reports

        """
        pass

    @abstractmethod
    def run(self, columns=[]) -> pd.DataFrame:
        """

        Runs check on provided columns

        :param columns:
        :return: CheckResult

        """
        pass

    def needed_preprocessing(self) -> dict:
        """

        :return: Name of the Check

        """
        return dict()

    def set_data(self, data: pd.DataFrame):
        """

        Set the data the check will run on

        :param data: pd.DataFrame

        """
        self.data = data


class Report(metaclass=ABCMeta):

    def __init__(self, result={}):
        self.result = result

    @abstractmethod
    def print_report(self):
        """

        Print report for checked columns

        """

class Reports():

    def __init__(self, check_result, report_class):
        self.check_result = check_result
        self.result_class = report_class
        self.reports = []
        self.evaluate()

    def evaluate(self, **kwargs):
        self.reports.append(self.result_class(data=self.check_result, **kwargs))


## Deprecated
class CheckResult(metaclass=ABCMeta):

    def __init__(self, result={}):
        self.result = result
        raise NotImplementedError("Deprecated: Use Report instead")

    @abstractmethod
    def print_report(self):
        """

        Print report for checked columns

        """
