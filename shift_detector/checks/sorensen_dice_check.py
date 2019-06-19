from shift_detector.checks.check import Check, Report
from shift_detector.precalculations.n_gram import NGramType
from shift_detector.precalculations.sorensen_dice_precalculation import SorensenDicePrecalculations
from IPython.display import display

import pandas as pd


class SorensenDiceCheck(Check):

    def __init__(self, ngram_type=NGramType.character, n=3):
        self.ngram_type = ngram_type
        self.n = n

    def run(self, store):
        data = store[SorensenDicePrecalculations(ngram_type=self.ngram_type, n=self.n)]

        examined_columns = set(data.keys())
        shifted_columns = set()

        for column_name in data:
            baseline1, baseline2, distance = data[column_name]

            if (abs(baseline1 - baseline2) > 0.1 or
                    distance - baseline1 < 0.1 or
                    distance - baseline2 < 0.1):
                shifted_columns.add(column_name)

        return SorensenDiceReport("Sorensen Dice Check", examined_columns, shifted_columns, information=data)


class SorensenDiceReport(Report):

    def print_information(self):
        result_df = pd.DataFrame.from_dict(self.information, orient='index')
        result_df.columns=['Baseline in Dataset 1', 'Baseline in Dataset 2', 'Distance between Datasets']
        display(result_df)
