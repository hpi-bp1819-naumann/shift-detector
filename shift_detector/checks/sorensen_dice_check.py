from shift_detector.checks.check import Check, Report
from shift_detector.precalculations.n_gram import NGramType
from shift_detector.precalculations.sorensen_dice_precalculation import SorensenDicePrecalculations
from IPython.display import display

import pandas as pd


class SorensenDiceCheck(Check):

    def __init__(self, ngram_type=NGramType.character, n=3, threshold=0.1):
        self.ngram_type = ngram_type
        self.n = n
        self.threshold = threshold

    def run(self, store):
        data = store[SorensenDicePrecalculations(ngram_type=self.ngram_type, n=self.n)]

        examined_columns = set(data.keys())
        shifted_columns = set()

        for column_name in data:
            baseline1, baseline2, distance = data[column_name]

            if (abs(baseline1 - baseline2) > self.threshold or
                    distance - baseline1 < self.threshold or
                    distance - baseline2 < self.threshold):
                shifted_columns.add(column_name)

        return SorensenDiceReport("Sorensen Dice Check", examined_columns, shifted_columns,
                                  information=(self.threshold, data))


class SorensenDiceReport(Report):

    def print_information(self):
        result_df = pd.DataFrame.from_dict(self.information[1], orient='index')
        result_df.columns = ['Similarity within Dataset 1', 'Similarity within Dataset 2',
                             'Similarity between Datasets']
        result_df['Threshold'] = self.information[0]
        display(result_df.loc[self.shifted_columns])
