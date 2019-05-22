import pandas as pd
import numpy as np
from scipy import stats
from datawig.utils import random_split
from shift_detector.checks.Check import Check, Report
from shift_detector.preprocessors.DefaultEmbedding import DefaultEmbedding
from shift_detector.preprocessors.WordEmbeddings import WordEmbedding, EmbeddingType
from shift_detector.Utils import ColumnType
from gensim.models import FastText


## TODO: think about whether the specific result should store the data at all or will always be
##       passed in the print report header or the result will be only calculated once when
##       created
class Chi2Report(Report):
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
        print("Columns with a Shift (significance: {})".format(self.significance),
              self.remarkable_columns())


class Chi2Check(Check):
    
    def __init__(self, text_embedding=EmbeddingType.FastText, trained_text_embedding=None,
                categorical_threshold=100):
        """
        :param text_embedding:  Either a EmbeddingType or model class that has the methods
                                'build_vocab' and 'train'
        :param trained_text_embedding: Pretrained Model
        :param categorical_threshold: #TODO
        :param significance:    The chi2 value that needs to be exceeded in order to have to
                                similiar data sets.
        """
        super().__init__()
        '''
        self.text_embedding = WordEmbedding(model=text_embedding, trained_model=trained_text_embedding)
        '''
        self.categorical_threshold = categorical_threshold

    @staticmethod
    def name() -> str:
        return "Chi Squared"

    @staticmethod
    def report_class():
        return Chi2Report

    def needed_preprocessing(self) -> dict:
        return {
            ColumnType.categorical: DefaultEmbedding()
        }

    def run(self, columns=[]):
        result = pd.DataFrame()
        for df in self.data[ColumnType.categorical]:
            p1, p2 = random_split(df)
            column_statistics = self.column_statistics(p1, p2,
                                categorical_threshold=self.categorical_threshold)
            result = result.append(column_statistics, ignore_index=True)

        column_statistics = self.column_statistics(self.data[ColumnType.categorical][0],
                            self.data[ColumnType.categorical][1],
                            categorical_threshold=self.categorical_threshold)
        result = result.append(column_statistics, ignore_index=True)
        return result

    # Internal calculations

    def column_statistics(self, first_df, second_df, columns=[], categorical_threshold=100):
        c_stats = pd.DataFrame()
        if not columns:
            columns = list(first_df.columns)
        for column in columns:
            a_series = first_df[column]
            b_series = second_df[column]
            c_stats[column] = [self.chi_test(a_series, b_series)]
        return c_stats

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
        observed = pd.DataFrame.from_dict({'a': a_counts, 'b': b_counts})
        _, p, _, _ = stats.chi2_contingency(observed)
        return p
