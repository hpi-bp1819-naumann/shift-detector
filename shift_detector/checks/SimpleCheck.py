import pandas as pd
import pprint as pp

from shift_detector.checks.Check import Check, Report
from shift_detector.preprocessors.DefaultEmbedding import DefaultEmbedding
from copy import deepcopy


class SimpleCheckReport(Report):
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.metrics_thresholds_percentage = {'mean': 10, 'median': 10, 'min': 15, 'max': 15, 'quartile_1': 15,
                                              'quartile_3': 15, 'uniqueness': 10, 'distinctness': 10,
                                              'completeness': 10, 'std': 10}
        self.categorical_threshold = 0.05

    def relative_metric_difference(self, column, metric_name):
        metric_in_df1 = self.data['numerical'][column][metric_name]['df1']
        metric_in_df2 = self.data['numerical'][column][metric_name]['df2']

        if metric_in_df1 == 0 and metric_in_df2 == 0:
            return 0
        elif metric_in_df1 == 0:
            print('column', column, '\t \t', metric_name, ': no comparison of distance possible, division by zero')
            return

        relative_difference = (metric_in_df2 / metric_in_df1 - 1) * 100
        if metric_name in ['uniqueness', 'completeness', 'completeness']:
            relative_difference = metric_in_df2 - metric_in_df1

        return relative_difference

    @staticmethod
    def difference_to_string(metrics_difference):
        metrics_difference_string = str(metrics_difference) + ' %'
        if metrics_difference > 0:
            metrics_difference_string = '+' + metrics_difference_string

        return metrics_difference_string

    def print_numerical_report(self):
        numerical_comparison = self.data['numerical']
        for column_name, metrics in numerical_comparison.items():

            if 'df1' in numerical_comparison[column_name]['available_in'] and \
                    'df2' not in numerical_comparison[column_name]['available_in']:
                print('Column', column_name, 'not available in df2')

            elif 'df2' in numerical_comparison[column_name]['available_in'] and \
                    'df1' not in numerical_comparison[column_name]['available_in']:
                print('Column', column_name, 'not available in df1')
            else:
                for metric in metrics:
                    if metric == 'available_in':
                        continue

                    diff = self.relative_metric_difference(column_name, metric)
                    if diff is not None:
                        if abs(diff) > self.metrics_thresholds_percentage[metric]:
                            print('shift in column', column_name, '\t', metric, self.difference_to_string(diff))

    def print_categorical_report(self):
        categorical_comparison = self.data['categorical']
        for column_name, attribute in categorical_comparison.items():
            for attribute_name, attribute_values in attribute.items():

                if 'df1' not in attribute_values:
                    attribute_values['df1'] = 0

                if 'df2' not in attribute_values:
                    attribute_values['df2'] = 0

                diff = attribute_values['df1'] - attribute_values['df2']
                if diff > self.categorical_threshold:
                    print('shift in column ', column_name, 'attribute ', attribute_name, ': ', diff)

    def print_report(self):
        self.print_numerical_report()
        self.print_categorical_report()


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
        df1_numerical = self.data["int"][0]
        df2_numerical = self.data["int"][1]
        df1_categorical = self.data["category"][0]
        df2_categorical = self.data['category'][1]

        numerical_comparison = self.compare_numerical_columns(df1_numerical, df2_numerical)
        categorical_comparison = self.compare_categorical_columns(df1_categorical, df2_categorical)

        return {'numerical': numerical_comparison, 'categorical': categorical_comparison}

    @staticmethod
    def compare_numerical_columns(df1, df2):
        numerical_comparison = dict()
        empty_metrics_dict = {'mean': {}, 'median': {}, 'min': {}, 'max': {}, 'quartile_1': {}, 'quartile_3': {},
                              'uniqueness': {}, 'distinctness': {}, 'completeness': {}, 'std': {}, 'available_in': {}}

        for df_name, df in [('df1', df1), ('df2', df2)]:
            for column in df.columns:
                if df_name == 'df1':
                    numerical_comparison[column] = deepcopy(empty_metrics_dict)
                elif not numerical_comparison.get(column):
                    numerical_comparison[column] = deepcopy(empty_metrics_dict)

                numerical_comparison[column]['available_in'][df_name] = True

                # TODO Later Vielleicht: verschnellerbar, in dem man alle Quantile gleichzeitig berechnet,
                #  also quantile([0, 0.25,  ... ]) oder Methoden selbst berechnet
                numerical_comparison[column]['min'][df_name] = df[column].min()
                numerical_comparison[column]['max'][df_name] = df[column].max()
                numerical_comparison[column]['quartile_1'][df_name] = df[column].quantile(.25)
                numerical_comparison[column]['quartile_3'][df_name] = df[column].quantile(.75)
                numerical_comparison[column]['median'][df_name] = df[column].median()

                numerical_comparison[column]['mean'][df_name] = df[column].mean()
                numerical_comparison[column]['std'][df_name] = df[column].std()

                numerical_comparison[column]['distinctness'][df_name] = df[column].nunique() / len(df1[column])
                numerical_comparison[column]['completeness'][df_name] = df[column].count() / len(df1[column])
                numerical_comparison[column]['uniqueness'][df_name] = len(df.groupby(column)
                                                                    .filter(lambda x: len(x) == 1)) / len(df1[column])
        return numerical_comparison

    @staticmethod
    def compare_categorical_columns(df1, df2):
        category_comparison = {}
        for column in df1.columns:
            category_comparison[column] = {}
            attribute_ratios = df1[column].value_counts(normalize=True).to_dict()
            for key, value in attribute_ratios.items():
                category_comparison[column][key] = {}
                category_comparison[column][key]['df1'] = value

        for column in df2.columns:
            if not category_comparison.get(column):
                category_comparison[column] = {}

            attribute_ratios = df2[column].value_counts(normalize=True).to_dict()
            for key, value in attribute_ratios.items():
                if key not in category_comparison[column]:
                    category_comparison[column][key] = {}
                category_comparison[column][key]['df2'] = value

        return category_comparison
