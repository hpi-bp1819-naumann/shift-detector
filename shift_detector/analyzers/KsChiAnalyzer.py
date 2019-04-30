import pandas as pd
import numpy as np
from scipy import stats
from datawig.utils import random_split
from shift_detector.analyzers.analyzer import Analyzer, AnalyzerResult


class KsChiResult(AnalyzerResult):

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


class KsChiAnalyzer(Analyzer):

    def __init__(self, first_df, second_df):
        Analyzer.__init__(self, first_df, second_df)
        
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
        chi2, p, dof, expected = stats.chi2_contingency(observed)
        return p

    # kolmogorov-smirnov
    def ks_test(self, a_series, b_series):
        ks_test_result = stats.ks_2samp(a_series, b_series)
        return ks_test_result.pvalue

    def column_statistics(self, first_df, second_df, columns=[], categorical_threshold=100):
        c_stats = pd.DataFrame()
        if not columns:
            columns = list(first_df.columns)
        for column in columns:
            a_series = first_df[column]
            b_series = second_df[column]
            if a_series.dtype in [np.float64, np.int64]:
                c_stats[column] = [self.ks_test(a_series, b_series)]
            else:#elif len(a_series.unique()) <= categorical_threshold:
                c_stats[column] = [self.chi_test(a_series, b_series)]
            #else:
                # treat as text
                #c_stats
        return c_stats

    def run(self, columns=[]):
        results = pd.DataFrame()
        for df in [self.first_df, self.second_df]:
            p1, p2 = random_split(df)
            results = results.append(self.column_statistics(p1, p2, columns), ignore_index=True)
        results = results.append(self.column_statistics(self.first_df, self.second_df, columns), ignore_index=True)
        return KsChiResult(results)
