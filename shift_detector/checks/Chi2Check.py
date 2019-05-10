import pandas as pd
import numpy as np
from scipy import stats
from datawig.utils import random_split
from shift_detector.checks.Check import Check, CheckResult
from shift_detector.preprocessors.Default import Default
from shift_detector.preprocessors.WordEmbeddings import WordEmbedding, EmbeddingType
from gensim.models import FastText

class Chi2Result(CheckResult):

    def __init__(self, data, significance=0.01):
        self.data = data
        self.significance = significance
    
    def remarkable_columns(self):
        # return names of columns for which inner set test didn't fail, but cross set test failed
        return list(self.data[self.data.columns[
            (self.data.loc[0] >= self.significance) & 
            (self.data.loc[1] >= self.significance) & 
            (self.data.loc[2] < self.significance)
        ]])

    def pvalues(self):
        return self.data.loc[2]

    def failing_feature_ratio(self):
        return len(self.data.loc[2][self.data.loc[2] < self.significance]) / len(self.data.columns)
    
    def print_report(self):
        """

        Print report for analyzed columns

        """
        print('Columns with a Shift:', self.remarkable_columns())
    

class Chi2Check(Check):
    # TODO: remove first and second df
    def __init__(self, first_df, second_df):
        Check.__init__(self, first_df, second_df)
        self.data = dict()

    def set_data(self, data):
        self.data = data

    @staticmethod
    def needed_preprocessing():
        return {
            "category": Default(),
            "text": WordEmbedding(model=FastText(size=200, window=5, min_count=1, workers=4))
        }

    # chi-squared
    def chi_test(self, a_series, b_series):
        a_counts = a_series.value_counts()
        b_counts = b_series.value_counts()
        for value in a_counts.index:
            if value not in b_counts:
                b_counts = b_counts.append(pd.Series(0, index=[value]))
        for value in b_counts.index:
            if value not in a_counts:
                a_counts = a_counts.append(pd.Series(0, index=[value]))
        observed = pd.DataFrame.from_dict({'a':a_counts, 'b':b_counts})
        _, p, _, _ = stats.chi2_contingency(observed)
        return p

    def column_statistics(self, first_df, second_df, columns=[], categorical_threshold=100):
        c_stats = pd.DataFrame()
        if not columns:
            columns = list(first_df.columns)
        for column in columns:
            a_series = first_df[column]
            b_series = second_df[column]
            c_stats[column] = [self.chi_test(a_series, b_series)]
        return c_stats

    def run(self, columns=[]):
        results = pd.DataFrame()
        for df in self.data["category"]:
            p1, p2 = random_split(df)
            results = results.append(self.column_statistics(p1, p2), ignore_index=True)

        results = results.append(self.column_statistics(self.data["category"][0], self.data["category"][1]), ignore_index=True)
        return Chi2Result(results)