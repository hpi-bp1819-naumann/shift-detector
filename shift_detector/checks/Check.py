from abc import ABCMeta, abstractmethod


class Report(metaclass=ABCMeta):

    @abstractmethod
    def print_report(self):
        """
        Print the report.
        """


class Reports:

    def __init__(self, check_result, report_class):
        self.check_result = check_result
        self.result_class = report_class
        self.reports = []
        self.evaluate()

    def evaluate(self, **kwargs):
        self.reports.append(self.result_class(data=self.check_result, **kwargs))


class Check(metaclass=ABCMeta):

    @abstractmethod
    def run(self, store) -> Report:
        """
        Run the check.
        :param store:
        :return: Report
        """
        pass
