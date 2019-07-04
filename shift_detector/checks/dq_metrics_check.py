import logging as logger
from collections import defaultdict, namedtuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from shift_detector.checks.check import Check, Report
from shift_detector.precalculations.dq_metrics_precalculation import DQMetricsPrecalculation
from shift_detector.utils.column_management import ColumnType
from shift_detector.utils.custom_print import nprint, diagram_title, dataset_names

ReportRow = namedtuple('ReportRow', 'metric_name val1 val2 threshold diff')


class DQMetricsCheck(Check):

    def __init__(self, categorical_threshold=0.05, mean_threshold=0.15, median_threshold=0.15,
                 value_range_threshold=0.5, quartile_1_threshold=0.2, quartile_3_threshold=0.2,
                 uniqueness_threshold=0.1, num_distinct_threshold=0.2, completeness_threshold=0.1, std_threshold=0.25):

        threshold_names_values = {'mean': mean_threshold, 'median': median_threshold,
                                  'value_range': value_range_threshold, 'quartile_1': quartile_1_threshold,
                                  'quartile_3': quartile_3_threshold, 'uniqueness': uniqueness_threshold,
                                  'num_distinct': num_distinct_threshold, 'std': std_threshold, 'completeness':
                                  completeness_threshold}

        if categorical_threshold < 0:
            raise ValueError("The categorical threshold of {} is not correct. It has to be larger than 0.0"
                             .format(categorical_threshold))

        for t_name, t_value in threshold_names_values.items():
            if t_value < 0:
                raise ValueError("The {}_threshold of {} is not correct. It has to be larger than 0.0"
                                 .format(t_name, t_value))

        self.data = None
        self.categorical_threshold = categorical_threshold
        self.metrics_thresholds_percentage = threshold_names_values

    def run(self, store):
        df1_numerical, df2_numerical = store[ColumnType.numerical]
        self.data = store[DQMetricsPrecalculation()]

        numerical_report = self.numerical_categorical_report(df1_numerical, df2_numerical)
        attribute_val_report = self.attribute_val_report()

        both_explanations = numerical_report.explanation.copy()
        both_explanations.update(attribute_val_report.explanation)
        both_reports = DQMetricsReport(set(numerical_report.examined_columns + attribute_val_report.examined_columns),
                                       set(numerical_report.shifted_columns + attribute_val_report.shifted_columns),
                                       both_explanations, {},
                                       numerical_report.figures + attribute_val_report.figures)

        return both_reports

    def relative_metric_difference(self, column, metric_name, comparison='numerical_comparison'):
        metric_in_df1 = self.data[comparison][column][metric_name]['df1']
        metric_in_df2 = self.data[comparison][column][metric_name]['df2']
        relative_difference = 0

        if metric_name in ['uniqueness', 'completeness']:
            relative_difference = metric_in_df2 - metric_in_df1

        elif metric_in_df1 == 0:
            logger.warning("Column {} \t \t {}: no comparison of distance possible, division by zero"
                           .format(column, metric_name))
        else:
            relative_difference = (metric_in_df2 / metric_in_df1 - 1)

        return metric_in_df1, metric_in_df2, relative_difference

    def numerical_categorical_report(self, df1, df2):
        numerical_comparison = self.data['numerical_comparison']
        categorical_comparison = self.data['categorical_comparison']

        examined_columns = set()
        shifted_columns = set()
        explanation = defaultdict(list)

        for comparison, comparison_name in [(numerical_comparison, 'numerical_comparison'), (categorical_comparison,
                                                                                             'categorical_comparison')]:
            for column_name, metrics in sorted(comparison.items()):
                examined_columns.add(column_name)

                for metric_name in metrics:
                    val1, val2, diff = self.relative_metric_difference(column_name, metric_name,
                                                                       comparison=comparison_name)
                    if abs(diff) > self.metrics_thresholds_percentage[metric_name]:
                        shifted_columns.add(column_name)

                        explanation[column_name].append(
                            ReportRow(metric_name, val1, val2, self.metrics_thresholds_percentage[metric_name], diff))

        return DQMetricsReport(examined_columns, shifted_columns, explanation={'numerical_categorical': explanation},
                               figures=[DQMetricsReport.numerical_plot(df1, df2)])

    def attribute_val_report(self):
        attribute_val_comparison = self.data['attribute_val_comparison']
        examined_columns = set()
        shifted_columns = set()
        explanation = defaultdict(list)
        plot_infos = []

        for column_name, attribute in sorted(attribute_val_comparison.items()):
            examined_columns.add(column_name)

            bar_df1 = []
            bar_df2 = []
            attribute_names = []

            for attribute_name, attribute_values in attribute.items():
                val1 = attribute_values['df1']
                val2 = attribute_values['df2']
                diff = val1 - val2

                bar_df1.append(val1)
                bar_df2.append(val2)
                attribute_names.append(attribute_name)

                if diff > self.categorical_threshold:
                    shifted_columns.add(column_name)
                    explanation[column_name].append(
                        ReportRow(attribute_name, val1, val2, self.categorical_threshold, diff))

            plot_infos.append((bar_df1, bar_df2, attribute_names, column_name))

        return DQMetricsReport(examined_columns, shifted_columns, explanation={'attribute_val': explanation},
                               information={}, figures=[DQMetricsReport.attribute_val_plot(plot_infos)])


class DQMetricsReport(Report):

    def __init__(self, examined_columns, shifted_columns, explanation=[], information={}, figures=[]):
        super().__init__("DQ Metrics Check", examined_columns, shifted_columns, explanation, information, figures)

    def print_explanation(self):

        for metric_name, name, sub_name, heading in \
                [('numerical_categorical', 'Metric', 'Column', 'Numerical Columns'),
                 ('attribute_val', 'Attribute Value', 'Attribute', 'Categorical Columns')]:

            nprint(heading, text_formatting='h3')
            for column_name, data in self.explanation[metric_name].items():
                names = []
                val1s = []
                val2s = []
                thresholds = []
                diffs = []

                for row in data:
                    names.append(row.metric_name)
                    val1s.append(round(row.val1, 2))
                    val2s.append(round(row.val2, 2))
                    thresholds.append(self.difference_to_string(row.threshold, print_plus_minus=True))
                    diffs.append(self.difference_to_string(row.diff))

                table_data = {name: names, 'Val in DS1': val1s, 'Val in DS2': val2s, 'Threshold': thresholds,
                              'Relative Diff': diffs}
                table = pd.DataFrame(table_data)

                print("{} '{}':".format(sub_name, column_name))
                nprint(table)
                print("\n")

    @staticmethod
    def difference_to_string(metrics_difference, print_plus_minus=False):
        metrics_difference = round(metrics_difference * 100, 2)
        metrics_difference_string = str(metrics_difference) + ' %'

        if print_plus_minus:
            metrics_difference_string = '+/- ' + metrics_difference_string
        elif metrics_difference > 0:
            metrics_difference_string = '+' + metrics_difference_string

        return metrics_difference_string

    @staticmethod
    def numerical_plot(df1, df2):
        def custom_plot():
            num_figures = len(list(df1.columns))
            num_cols = 5

            f = plt.figure()
            f.set_figheight(6 * (num_figures/num_cols))
            f.set_figwidth(20)

            for num, column in enumerate(sorted(list(df1.columns))):
                a, b = df1[column].dropna(), df2[column].dropna()
                ax = f.add_subplot(num_figures/num_cols + 1, num_cols, num + 1)

                ax.boxplot([a, b], labels=[dataset_names()[0], dataset_names()[1]])
                ax.set_title(diagram_title(column))

            plt.show()

        return custom_plot

    @staticmethod
    def attribute_val_plot(plot_infos):
        def custom_plot():
            num_figures = len(list(plot_infos))
            num_cols = 3

            f = plt.figure()
            f.set_figheight(8 * (num_figures/num_cols))
            f.set_figwidth(18)

            for i, plot_info in enumerate(list(plot_infos)):
                bars1, bars2, attribute_names, column_name = plot_info[0], plot_info[1], plot_info[2], plot_info[3]

                subplot = f.add_subplot(len(list(plot_infos))/3 + 1, 3, i + 1)
                bar_width = 0.25

                ind = np.arange(len(bars1))
                subplot.barh(ind + bar_width, bars1, bar_width, label=dataset_names()[0])
                subplot.barh(ind, bars2, bar_width, label=dataset_names()[1])

                subplot.set_yticklabels(attribute_names)
                subplot.set_yticks(np.arange(len(attribute_names)) + bar_width / 2)

                subplot.title.set_text(diagram_title(column_name))
                subplot.legend()

            plt.show()

        return custom_plot
