from abc import ABCMeta, abstractmethod
from collections import defaultdict
from itertools import chain


class Report:

    def __init__(self, examined_columns, shifted_columns, explanation=dict(), information=dict()):
        self.examined_columns = examined_columns
        self.shifted_columns = shifted_columns
        self.explanation = explanation
        self.information = information

    def __add__(self, other):
        if not isinstance(other, Report):
            raise Exception('Tried to add class of type {} to Report'.format(other.__class__))

        self.examined_columns = set(self.examined_columns).union(other.examined_columns)
        self.shifted_columns = set(self.shifted_columns).union(other.shifted_columns)

        self.explanation = self.__sum_dicts(self.explanation, other.explanation)
        self.information = self.__sum_dicts(self.information, other.information)

        return self

    @staticmethod
    def __sum_dicts(dict1, dict2):
        res_dict = defaultdict(str)

        for key, value in chain(dict1.items(), dict2.items()):
            if key in res_dict:
                res_dict[key] += '\n'
            res_dict[key] += value

        return res_dict

    def __str__(self):
        msg = ""
        msg += "Examined Columns: {}\n".format(self.examined_columns)
        msg += "Shifted Columns: {}\n\n".format(self.shifted_columns)

        for column, explanation in self.explanation.items():
            msg += "Column '{}':\n{}\n".format(column, explanation)

        for tag, information in self.information.items():
            msg += "'{}':\n{}\n".format(tag, information)

        return msg


class DeprecatedReport(metaclass=ABCMeta):

    @abstractmethod
    def print_report(self):
        """
        Print the report.
        """
        pass


class Check(metaclass=ABCMeta):

    @abstractmethod
    def run(self, store) -> Report:
        """
        Run the check.
        :param store:
        :return: Report
        """
        pass
