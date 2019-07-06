import re
from unittest import TestCase
import mock

from pandas import DataFrame

from shift_detector.checks.dq_metrics_check import DQMetricsCheck
from shift_detector.precalculations.store import Store


class TestSimpleCheck(TestCase):

    def setUp(self):
        sales1 = {'shift': ['A'] * 100, 'no_shift': ['C'] * 100}
        sales2 = {'shift': ['B'] * 100, 'no_shift': ['C'] * 100}
        numbers1 = {'cool_numbers': [1, 2, 3, 4] * 10}
        numbers2 = {'cool_numbers': [1, 2, 3, 6] * 10}

        self.df1 = DataFrame.from_dict(sales1)
        self.df2 = DataFrame.from_dict(sales2)
        self.df1_num = DataFrame.from_dict(numbers1)
        self.df2_num = DataFrame.from_dict(numbers2)

        self.store = Store(self.df1, self.df2)
        self.store_num = Store(self.df1_num, self.df2_num)
        self.check = DQMetricsCheck()

    def test_init(self):
        with self.subTest('normal analyzers should detect shift'):
            self.check = DQMetricsCheck()
            report = self.check.run(self.store_num)

            explanations = report.explanation['numerical_categorical']['cool_numbers']

            self.assertEqual(explanations[0].val1, 2.5)
            self.assertEqual(explanations[0].val2, 3.0)
            self.assertEqual(round(explanations[0].diff, 2), 0.2)

            self.assertEqual(explanations[1].val1, 3.0)
            self.assertEqual(explanations[1].val2, 5.0)
            self.assertEqual(round(explanations[1].diff, 2), 0.67)

            self.assertEqual(round(explanations[2].val1, 2), 1.13)
            self.assertEqual(round(explanations[2].val2, 2), 1.89)
            self.assertEqual(round(explanations[2].diff, 2), 0.67)

        with self.subTest('no analyzer should detect shift'):
            self.check = DQMetricsCheck(mean_threshold=.3, value_range_threshold=.68, std_threshold=.7,
                                        categorical_threshold=.25)
            report = self.check.run(self.store_num)
            self.assertEqual(report.shifted_columns, [])

        with self.subTest('only std should detect shift'):
            self.check = DQMetricsCheck(mean_threshold=.3, value_range_threshold=.68, std_threshold=.5,
                                        categorical_threshold=.25)

            report = self.check.run(self.store_num)
            explanations = report.explanation['numerical_categorical']['cool_numbers']
            self.assertEqual(len(explanations), 1)
            self.assertEqual(round(explanations[0].val1, 2), 1.13)
            self.assertEqual(round(explanations[0].val2, 2), 1.89)
            self.assertEqual(round(explanations[0].diff, 2), 0.67)

    def test_run_categorical(self):
        with self.subTest("Test precalculation"):
            report = self.check.run(self.store)

        with self.subTest("Test shifted categorical columns"):
            self.assertEqual(report.shifted_columns, ['shift'])
            self.assertCountEqual(report.examined_columns, ['shift', 'no_shift'])

            shift_explanation = report.explanation['attribute_val']['shift'][0]
            self.assertEqual(shift_explanation.val1, 1.0)
            self.assertEqual(shift_explanation.val2, 0)
            self.assertEqual(shift_explanation.diff, 1.0)

    def test_relative_metric_difference(self):
        with self.subTest('Normal case'):
            self.check.data = {'numerical_comparison': {'column': {'metric_name': {'df1': 10.0, 'df2': 5.0}}}}
            self.assertEqual(self.check.relative_metric_difference('column', 'metric_name')[2], -.5)

            self.check.data = {'numerical_comparison': {'column': {'metric_name': {'df1': 1.2, 'df2': 12.0}}}}
            self.assertEqual(self.check.relative_metric_difference('column', 'metric_name')[2], 9.0)

        with self.subTest('Both values zero'):
            self.check.data = {'numerical_comparison': {'column': {'metric_name': {'df1': 0, 'df2': 0}}}}
            self.assertEqual(self.check.relative_metric_difference('column', 'metric_name')[2], 0)

        with self.subTest('Value in df1 zero'):
            self.check.data = {'numerical_comparison': {'column': {'metric_name': {'df1': 0, 'df2': 12.4}}}}
            self.assertEqual(self.check.relative_metric_difference('column', 'metric_name')[2], 0)

    @mock.patch('shift_detector.checks.dq_metrics_check.plt')
    def test_numerical_plots_work(self, mock_plt):
        self.assertFalse(mock_plt.figure.called)

        report = self.check.run(self.store)
        custom_plot_numerical = report.numerical_plot(DataFrame([1, 2, 3]), DataFrame([4, 5, 6]))
        custom_plot_numerical()

        self.assertTrue(mock_plt.figure.called)
        self.assertTrue(mock_plt.figure().add_subplot.called)
        self.assertTrue(mock_plt.show.called)

    @mock.patch('shift_detector.checks.dq_metrics_check.plt')
    def test_categorical_plots_work(self, mock_plt):
        self.assertFalse(mock_plt.figure.called)

        report = self.check.run(self.store)
        custom_plot_categorical = report.attribute_val_plot([([1, 2, 3], [2, 4, 6], ['Heinz', 'Peter', 'Rudolf'],
                                                            'A very important plot')])
        custom_plot_categorical()

        self.assertTrue(mock_plt.figure.called)
        self.assertTrue(mock_plt.figure().add_subplot.called)
        self.assertTrue(mock_plt.show.called)
