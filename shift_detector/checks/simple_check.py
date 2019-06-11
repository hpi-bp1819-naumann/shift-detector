import logging as logger
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from shift_detector.checks.Check import Check, Report
from shift_detector.precalculations.SimplePrecalculation import SimplePrecalculation
from shift_detector.utils.ColumnManagement import ColumnType


class SimpleCheck(Check):

    def __init__(self):
        self.data = None
        self.categorical_threshold = 0.05
        self.metrics_thresholds_percentage = {'mean': 10, 'median': 10, 'min': 15, 'max': 15, 'quartile_1': 15,
                                              'quartile_3': 15, 'uniqueness': 10, 'num_distinct': 10,
                                              'completeness': 10, 'std': 10}

    def run(self, store):
        logger.info("Execute Simple Check")
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
        # TODO: think about comparison if base value is 0
        if metric_in_df1 == 0:
            logger.warning('column', column, '\t \t', metric_name,
                           ': no comparison of distance possible, division by zero')
            return 0

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

    def numerical_report(self, df1, df2):
        numerical_comparison = self.data['numerical_comparison']
        examined_columns = set()
        shifted_columns = set()
        explanation = defaultdict(str)

        for column_name, metrics in numerical_comparison.items():
            examined_columns.add(column_name)

            for metric in metrics:
                diff = self.relative_metric_difference(column_name, metric)

                if abs(diff) > self.metrics_thresholds_percentage[metric]:
                    shifted_columns.add(column_name)
                    explanation[column_name] += "Metric: {} with Diff: {}\n".format(metric,
                                                                                    self.difference_to_string(diff))

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

                if 'df1' not in attribute_values:
                    attribute_values['df1'] = 0

                if 'df2' not in attribute_values:
                    attribute_values['df2'] = 0

                diff = attribute_values['df1'] - attribute_values['df2']

                bar_df1.append(attribute_values['df1'])
                bar_df2.append(attribute_values['df2'])
                attribute_names.append(attribute_name)

                print('here we have value ', attribute_values['df1'], attribute_values['df2'], 'wich a diff in ',
                      column_name, ' in attribute ', attribute_name)

                if diff > self.categorical_threshold:
                    shifted_columns.add(column_name)
                    explanation[column_name] += "Attribute: {} with Diff: {}\n".format(attribute_name, diff)

            plot_infos.append((bar_df1, bar_df2, attribute_names, column_name))

        return SimpleReport(examined_columns, shifted_columns, dict(explanation),
                            figures=[SimpleReport.categorical_plot(plot_infos)])


class SimpleReport(Report):

    def __init__(self, examined_columns, shifted_columns, information={}, explanation={}, figures=[]):
        super().__init__("Simple Check", examined_columns, shifted_columns, information, explanation, figures)

    @staticmethod
    def numerical_plot(df1, df2):
        def custom_plot():
            f = plt.figure(figsize=(20, 7))
            num_columns = len(list(df1.columns))
            for num, column in enumerate(list(df1.columns)):
                a, b = df1[column], df2[column]
                ax = f.add_subplot(1, num_columns, num+1)

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

                subplot = f.add_subplot(1, num_columns, i+1)

                # set width of bar
                bar_width = 0.25

                # Set position of bar on X axis
                r1 = np.arange(len(bars1))
                r2 = [x + bar_width for x in r1]

                # Make the plot
                subplot.bar(r1, bars1, color='red', width=bar_width, edgecolor='white', label='DS1')
                subplot.bar(r2, bars2, color='blue', width=bar_width, edgecolor='white', label='DS2')

                # Add xticks on the middle of the group bars
                subplot.set_xlabel('attribute', fontweight='bold')
                subplot.set_xticks(np.arange(len(attribute_names))+bar_width/2)
                subplot.set_xticklabels(attribute_names)

                # Create legend & Show graphic
                subplot.legend()

            f.show()

        return custom_plot

