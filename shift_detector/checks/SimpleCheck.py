import pandas as pd
import pprint as pp

from shift_detector.checks.Check import Check, Report
from shift_detector.preprocessors.DefaultEmbedding import DefaultEmbedding
from copy import deepcopy

class SimpleCheckReport(Report):
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.metrics_thresholds_percentage = {'mean': 10, 'median': 10, 'min': 5, 'max': 10, 'quartile_1': 10,
                                              'quartile_3': 10, 'uniqueness': 10, 'distinctness': 10,
                                              'completeness': 10, 'std': 10}

    def relative_metric_difference(self, column, metric):
        metric_in_df1 = self.data[column][metric]['df1']
        metric_in_df2 = self.data[column][metric]['df2']

        if metric_in_df1 == 0:
            print('column', column, '\t \t metric', metric, ': no comparison of distance possible, division by zero')
            return

        relative_difference = (metric_in_df2 / metric_in_df1 - 1) * 100
        return relative_difference

    @staticmethod
    def difference_to_string(metrics_difference):
        metrics_difference_string = str(metrics_difference) + ' %'
        if metrics_difference > 0:
            metrics_difference_string = '+' + metrics_difference_string

        return metrics_difference_string

    def print_report(self):
        for column, metrics in self.data.items():

            if 'df1' in self.data[column]['available_in'] and 'df2' not in self.data[column]['available_in']:
                print('Column', column, 'not available in df2')
            elif 'df2' in self.data[column]['available_in'] and 'df1' not in self.data[column]['available_in']:
                print('Column', column, 'not available in df1')
            else:
                for metric in metrics:
                    if metric == 'available_in':
                        continue

                    diff = self.relative_metric_difference(column, metric)
                    if diff is not None:
                        if abs(diff) > self.metrics_thresholds_percentage[metric]:
                            print('shift in column', column, '\t', metric, self.difference_to_string(diff))


class SimpleCheck(Check):
    @staticmethod
    def report_class():
        return SimpleCheckReport

    @staticmethod
    def name() -> str:
        return 'SimpleCheck'

    def needed_preprocessing(self) -> dict:
        return {
            "category": DefaultEmbedding(),
            "int": DefaultEmbedding(),
        }

    def run(self, columns=[]):
        # df1 = self.data["int"][0]
        # df2 = self.data["int"][1]

        df1 = pd.DataFrame([100, 100])
        df2 = pd.DataFrame([105, 105])

        comparison_dict = dict()
        empty_metrics_dict = {'mean': {}, 'median': {}, 'min': {}, 'max': {}, 'quartile_1': {}, 'quartile_3': {},
                              'uniqueness': {}, 'distinctness': {}, 'completeness': {}, 'std': {}, 'available_in': {}}

        for column in df1.columns:
            comparison_dict[column] = deepcopy(empty_metrics_dict)
            comparison_dict[column]['available_in']['df1'] = True

            # Eventuell verschnellerbar, in dem man alle Quantile gleichzeitig berechnet, als quantile([0, 0.25,  ... ])
            comparison_dict[column]['min']['df1'] = df1[column].min()
            comparison_dict[column]['max']['df1'] = df1[column].max()
            comparison_dict[column]['quartile_1']['df1'] = df1[column].quantile(.25)
            comparison_dict[column]['quartile_3']['df1'] = df1[column].quantile(.75)
            comparison_dict[column]['median']['df1'] = df1[column].median()

            comparison_dict[column]['mean']['df1'] = df1[column].mean()
            comparison_dict[column]['std']['df1'] = df1[column].std()

            comparison_dict[column]['distinctness']['df1'] = df1[column].nunique() / len(df1[column])
            comparison_dict[column]['completeness']['df1'] = df1[column].count() / len(df1[column])
            comparison_dict[column]['uniqueness']['df1'] = len(df1.groupby(column).filter(lambda x: len(x) == 1)) /  \
                                                           len(df1[column])

        for column in df2:
            if not comparison_dict.get(column):
                comparison_dict[column] = empty_metrics_dict
            comparison_dict[column]['available_in']['df2'] = True

            # Eventuell verschnellerbar, in dem man alle quantile gleichzeitig berechnet, als quantile([0, 0.25,  ... ])
            comparison_dict[column]['min']['df2'] = df2[column].min()
            comparison_dict[column]['max']['df2'] = df2[column].max()
            comparison_dict[column]['quartile_1']['df2'] = df2[column].quantile(.25)
            comparison_dict[column]['quartile_3']['df2'] = df2[column].quantile(.75)
            comparison_dict[column]['median']['df2'] = df2[column].median()

            comparison_dict[column]['mean']['df2'] = df2[column].mean()
            comparison_dict[column]['std']['df2'] = df2[column].std()

            comparison_dict[column]['distinctness']['df2'] = df2[column].nunique() / len(df2[column])
            comparison_dict[column]['completeness']['df2'] = df2[column].count() / len(df2[column])
            comparison_dict[column]['uniqueness']['df2'] = len(df2.groupby(column).filter(lambda x: len(x) == 1)) / \
                                                           len(df2[column])

        return comparison_dict

