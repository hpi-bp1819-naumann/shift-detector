from abc import ABCMeta, abstractmethod


class Precalculation(metaclass=ABCMeta):

    @abstractmethod
    def __eq__(self, other):
        """Overrides the default implementation"""
        pass

    @abstractmethod
    def __hash__(self):
        """Overrides the default implementation"""
        pass

    @abstractmethod
    def process(self, store):
        pass


class Report:

    def __init__(self, examined_columns, shifted_columns, explanation):
        self.examined_columns = examined_columns
        self.shifted_columns = shifted_columns
        self.explanation = explanation

    def __add__(self, other):
        if not isinstance(other, Report):
            raise Exception('Comparison of Report and Non-Report type')

        self.examined_columns = set(self.examined_columns).union(other.examined_columns)
        self.shifted_columns = set(self.shifted_columns).union(other.shifted_columns)

        for key in set(self.explanation.keys()).union(other.explanation.keys()):
            if key in self.explanation and key in other.explanation:
                self.explanation[key] += "\n{}".format(other.explanation[key])
            elif key in other.explanation:
                self.explanation[key] = other.explanation[key]

        return self

    def __str__(self):
        msg = ""
        msg += "Examined Columns: {}\n".format(self.examined_columns)
        msg += "Shifted Columns: {}\n\n".format(self.shifted_columns)
        for column, explanation in self.explanation.items():
            msg += "Column '{}':\n{}\n".format(column, explanation)

        return msg


class DeprecatedReport(metaclass=ABCMeta):

    @abstractmethod
    def print_report(self):
        """
        Print the report.
        """
        pass


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
    def run(self, store) -> DeprecatedReport:
        """
        Run the check.
        :param store:
        :return: Report
        """
        pass
