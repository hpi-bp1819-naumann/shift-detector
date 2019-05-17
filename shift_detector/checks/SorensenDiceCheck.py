import pandas as pd
from shift_detector.preprocessors.NGram import NGram, NGramType
from shift_detector.Utils import ColumnType
from shift_detector.checks.Check import Check, CheckResult


class SorensenDiceResult(CheckResult):
    def print_report(self):
        print('Hello World!')


class SorensenDiceCheck(Check):

    @staticmethod
    def name() -> str:
        """
        :return: Name of the check
        """
        return "Sørensen Dice Coefficient"

    @staticmethod
    def report_class():
        """
        :return: The class that will be used for reports
        """
        return

    def run(self, columns=[]) -> pd.DataFrame:
        """
        Calculate the sorensen dice coefficient between two columns
        :param columns:
        :return: CheckResult
        """

        return pd.DataFrame([])

    def needed_preprocessing(self) -> dict:
        return {ColumnType.text: NGram(5, NGramType.character)}
