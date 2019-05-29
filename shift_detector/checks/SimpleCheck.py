from copy import deepcopy

from shift_detector.checks.Check import Check, Report
from shift_detector.utils.ColumnManagement import ColumnType


class SimpleCheckReport(Report):
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.metrics_thresholds_percentage = {'mean': 10, 'median': 10, 'min': 15, 'max': 15, 'quartile_1': 15,
                                              'quartile_3': 15, 'uniqueness': 10, 'distinctness': 10,
                                              'completeness': 10, 'std': 10}
        self.categorical_threshold = 0.05

    def relative_metric_difference(self, column, metric_name):
        metric_in_df1 = self.data['numerical_comparison'][column][metric_name]['df1']
        metric_in_df2 = self.data['numerical_comparison'][column][metric_name]['df2']

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
        numerical_comparison = self.data['numerical_comparison']
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
        categorical_comparison = self.data['numerical_comparison']
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
    def run(self, store):
        df1_numerical = store[ColumnType.numerical][0]
        df2_numerical = store[ColumnType.numerical][1]
        df1_categorical = store[ColumnType.categorical][0]
        df2_categorical = store[ColumnType.categorical][1]

        numerical_comparison = self.compare_numerical_columns(df1_numerical, df2_numerical)
        categorical_comparison = self.compare_categorical_columns(df1_categorical, df2_categorical, df1_categorical.columns)
        combined_comparisons = {'categorical_comparison': categorical_comparison,
                                'numerical_comparison': numerical_comparison}
        return SimpleCheckReport(data=combined_comparisons)

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

                column_droppedna = df[column].dropna()
                numerical_comparison[column]['std'][df_name] = column_droppedna.std()

                numerical_comparison[column]['distinctness'][df_name] = column_droppedna.nunique() / \
                                                                        len(column_droppedna)

                numerical_comparison[column]['completeness'][df_name] = len(column_droppedna) / len(df1[column])

                numerical_comparison[column]['uniqueness'][df_name] = len(df.groupby(column)
                                                                    .filter(lambda x: len(x) == 1)) / \
                                                                    len(column_droppedna)
        return numerical_comparison

    @staticmethod
    def compare_categorical_columns(df1, df2, columns):
        category_comparison = {}

        for column in columns:
            category_comparison[column] = {}
            attribute_ratios_df1 = df1[column].value_counts(normalize=True).to_dict()
            # category_comparison[column]['df1'] = {}
            # category_comparison[column]['df2'] = {}

            for key, value in attribute_ratios_df1.items():
                category_comparison[column][key] = {'df1': value}

            attribute_ratios_df2 = df2[column].value_counts(normalize=True).to_dict()
            for key, value in attribute_ratios_df2.items():
                if category_comparison[column].get(key):
                    category_comparison[column][key]['df2'] = value

        return category_comparison
