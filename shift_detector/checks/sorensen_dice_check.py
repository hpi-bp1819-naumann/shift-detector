import logging as logger
from collections import defaultdict

from shift_detector.checks.check import Check, Report
from shift_detector.precalculations.n_gram import NGramType
from shift_detector.precalculations.sorensen_dice_precalculation import SorensenDicePrecalculations


class SorensenDiceCheck(Check):

    def __init__(self, ngram_type=NGramType.character, n=3):
        self.ngram_type = ngram_type
        self.n = n

    def run(self, store):
        logger.info("Execute Sorensen Dice Check")
        data = store[SorensenDicePrecalculations(ngram_type=self.ngram_type, n=self.n)]

        examined_columns = set(data.keys())
        shifted_columns = set()
        explanation = defaultdict(str)

        for column_name in data:
            baseline1, baseline2, distance = data[column_name]
            explanation[column_name] = "\tBaseline in Dataset1: {}\n" \
                                       "\tBaseline in Dataset2: {}\n" \
                                       "\tSorensen Dice Coefficient between Datasets: {}"\
                .format(baseline1, baseline2, distance)

            if (abs(baseline1 - baseline2) > 0.1 or
                    distance - baseline1 < 0.1 or
                    distance - baseline2 < 0.1):
                shifted_columns.add(column_name)

        return Report("Sorensen Dice Check", examined_columns, shifted_columns, dict(explanation))
