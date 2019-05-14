from shift_detector.checks.Check import Check, Report
from shift_detector.preprocessors.DefaultEmbedding import DefaultEmbedding
import pandas as pd
from datawig.utils import random_split


class SimpleCheckReport(Report):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def print_report(self):
        pass


class SimpleCheck(Check):
    @staticmethod
    def report_class():
        return SimpleCheckReport

    @staticmethod
    def name() -> str:
        return 'Simple'

    def needed_preprocessing(self) -> dict:
        return {
            "category": DefaultEmbedding()
        }

    def run(self, columns=[]):
        return []

