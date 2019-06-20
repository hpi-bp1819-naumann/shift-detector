from abc import ABCMeta, abstractmethod
from collections import defaultdict
from itertools import chain

from shift_detector.utils.custom_print import nprint


class Report:

    def __init__(self, check_name, examined_columns, shifted_columns, explanation={}, information={}, figures=[]):
        self.check_name = check_name
        self.examined_columns = list(examined_columns)
        self.shifted_columns = list(shifted_columns)
        self.explanation = explanation
        self.information = information
        self.figures = figures

    def __add__(self, other):
        if not isinstance(other, Report):
            raise Exception('Tried to add class of type {} to Report'.format(other.__class__))

        self.examined_columns = list(set(self.examined_columns).union(other.examined_columns))
        self.shifted_columns = list(set(self.shifted_columns).union(other.shifted_columns))

        self.explanation = self.__sum_dicts(self.explanation, other.explanation)
        self.information = self.__sum_dicts(self.information, other.information)

        self.figures += other.figures

        return self

    @staticmethod
    def __sum_dicts(dict1, dict2):
        res_dict = defaultdict(str)

        for key, value in chain(dict1.items(), dict2.items()):
            if key in res_dict:
                res_dict[key] += '\n'
            res_dict[key] += value

        return res_dict

    def print_report(self):
        nprint(self.check_name, text_formatting='h2')
        print("Examined Columns: {}".format(self.examined_columns))
        print("Shifted Columns: {}".format(self.shifted_columns))
        print()

        self.print_explanation()
        self.print_information()

    def print_explanation(self):
        msg = ""
        for column, explanation in self.explanation.items():
            msg += "Column '{}':\n{}\n".format(column, explanation)
        print(msg)

    def print_information(self):
        msg = ""
        for tag, information in self.information.items():
            msg += "'{}':\n{}\n".format(tag, information)
        print(msg)


class Check(metaclass=ABCMeta):

    @abstractmethod
    def run(self, store):
        """
        Run the check.
        :param store:
        :return: Anything
        """
        pass
