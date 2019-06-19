import logging as logger
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from shift_detector.checks.check import Check, Report
from shift_detector.precalculations.simple_precalculation import SimplePrecalculation
from shift_detector.utils.column_management import ColumnType
from shift_detector.utils.neat_print import nprint


class SimpleCheck(Check):

    def __init__(self, categorical_threshold=0.05, mean_threshold=0.15, median_threshold=0.15,
                 value_range_threshold=0.5, quartile_1_threshold=0.2, quartile_3_threshold=0.2,
                 uniqueness_threshold=0.1, num_distinct_threshold=0.2, completeness_threshold=0.1, std_threshold=0.25):

        threshold_names_values = {'mean': mean_threshold, 'median': median_threshold,
                                  'value_range': value_range_threshold, 'quartile_1': quartile_1_threshold,
                                  'quartile_3': quartile_3_threshold, 'uniqueness': uniqueness_threshold,
                                  'num_distinct': num_distinct_threshold, 'std': std_threshold, 'completeness':
                                  completeness_threshold}

        if categorical_threshold < 0 or categorical_threshold > 1:
            raise ValueError('The categorical threshold of {} is not correct. It has be between the values of '
                             '0 and 1.'.format(categorical_threshold))

        for t_name, t_value in threshold_names_values.items():
            if t_value < 0 or t_value > 1:
                raise ValueError('The {}_threshold of {} is not correct. It has be between the values of 0 and 1'
                                 .format(t_name, t_value))

        self.data = None
        self.categorical_threshold = categorical_threshold
        self.metrics_thresholds_percentage = threshold_names_values

    def run(self, store):
        df1_numerical, df2_numerical = store[ColumnType.numerical]
        self.data = store[SimplePrecalculation()]
        numerical_report = self.numerical_report(df1_numerical, df2_numerical)
        categorical_report = self.categorical_report()

        return numerical_report + categorical_report

    def relative_metric_difference(self, column, metric_name):
        metric_in_df1 = self.data['numerical_comparison'][column][metric_name]['df1']
        metric_in_df2 = self.data['numerical_comparison'][column][metric_name]['df2']

        if metric_in_df1 == 0 and metric_in_df2 == 0:
            return 0
        if metric_in_df1 == 0:
            logger.warning('column {} \t \t {}: no comparison of distance possible, division by zero'
                           .format(column, metric_name))
            return 0

        relative_difference = (metric_in_df2 / metric_in_df1 - 1)
        if metric_name in ['uniqueness', 'completeness', 'completeness']:
            relative_difference = metric_in_df2 - metric_in_df1

        return relative_difference

    @staticmethod
    def difference_to_string(metrics_difference, threshold=False):
        metrics_difference = round(metrics_difference*100, 2)
        metrics_difference_string = str(metrics_difference) + ' %'

        if threshold:
            metrics_difference_string = '+/- ' + metrics_difference_string
        elif metrics_difference > 0:
            metrics_difference_string = '+' + metrics_difference_string

        return metrics_difference_string

    def numerical_report(self, df1, df2):
        numerical_comparison = self.data['numerical_comparison']
        examined_columns = set()
        shifted_columns = set()
        explanation = defaultdict(str)

        for column_name, metrics in numerical_comparison.items():
            examined_columns.add(column_name)

            print('column {}'.format(column_name))

            for metric in metrics:
                diff = self.relative_metric_difference(column_name, metric)
                if abs(diff) > self.metrics_thresholds_percentage[metric]:
                    shifted_columns.add(column_name)
                    explanation[column_name] += "Metric: {}, Diff: {}, threshold: {}\n".\
                        format(metric, self.difference_to_string(diff),
                               self.difference_to_string(self.metrics_thresholds_percentage[metric]), threshold=True)

        return SimpleReport(examined_columns, shifted_columns, dict(explanation),
                            figures=[SimpleReport.numerical_plot(df1, df2)])

    def categorical_report(self):
        categorical_comparison = self.data['categorical_comparison']
        examined_columns = set()
        shifted_columns = set()
        explanation = defaultdict(str)
        plot_infos = []

        for column_name, attribute in categorical_comparison.items():
            examined_columns.add(column_name)

            bar_df1 = []
            bar_df2 = []
            attribute_names = []

            for attribute_name, attribute_values in attribute.items():

                diff = attribute_values['df1'] - attribute_values['df2']

                bar_df1.append(attribute_values['df1'])
                bar_df2.append(attribute_values['df2'])
                attribute_names.append(attribute_name)

                if diff > self.categorical_threshold:
                    shifted_columns.add(column_name)
                    explanation[column_name] += "Attribute: '{}' with Diff: {}, categorical threshold: {}\n"\
                        .format(attribute_name, self.difference_to_string(diff),
                                self.difference_to_string(self.categorical_threshold))

            plot_infos.append((bar_df1, bar_df2, attribute_names, column_name))

        return SimpleReport(examined_columns, shifted_columns, dict(explanation),
                            figures=[SimpleReport.categorical_plot(plot_infos)])


class SimpleReport(Report):

    def __init__(self, examined_columns, shifted_columns, information={}, explanation={}, figures=[]):
        super().__init__("Simple Check", examined_columns, shifted_columns, information, explanation, figures)

    def print_explanation(self):
        categorical_found = False
        nprint("Numerical columns", text_formatting='h3')

        for column, explanation in self.explanation.items():
            if not categorical_found:
                if 'categorical' in explanation:
                    categorical_found = True
                    nprint("Categorical columns", text_formatting='h3')

            print("Column '{}':\n{}\n".format(column, explanation))

    @staticmethod
    def numerical_plot(df1, df2):
        def custom_plot():
            f = plt.figure(figsize=(20, 7))
            num_columns = len(list(df1.columns))
            for num, column in enumerate(list(df1.columns)):
                a, b = df1[column], df2[column]
                ax = f.add_subplot(1, num_columns, num + 1)

                ax.boxplot([a, b])
                ax.set_title(column)

            plt.show()

        return custom_plot

    @staticmethod
    def categorical_plot(plot_infos):

        def custom_plot():
            f = plt.figure(figsize=(20, 7))
            num_columns = len(list(plot_infos))

            for i, plot_info in enumerate(list(plot_infos)):
                bars1, bars2, attribute_names, column_name = plot_info[0], plot_info[1], plot_info[2], plot_info[3]

                subplot = f.add_subplot(1, num_columns, i + 1)

                bar_width = 0.25
                r1 = np.arange(len(bars1))
                r2 = [x + bar_width for x in r1]

                subplot.bar(r1, bars1, color='red', width=bar_width, edgecolor='white', label='DS1')
                subplot.bar(r2, bars2, color='blue', width=bar_width, edgecolor='white', label='DS2')

                subplot.title.set_text(column_name)
                subplot.set_xlabel('attribute-values', fontweight='bold')
                subplot.set_xticks(np.arange(len(attribute_names)) + bar_width / 2)
                subplot.set_xticklabels(attribute_names)
                subplot.legend()

            plt.show()

        return custom_plot
