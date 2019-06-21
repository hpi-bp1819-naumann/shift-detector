import re
from unittest import TestCase
import mock

from pandas import DataFrame

from shift_detector.checks.simple_check import SimpleCheck
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
        self.check = SimpleCheck()

    def test_init(self):
        with self.subTest('normal analyzers should detect shift'):
            self.check = SimpleCheck()
            report = self.check.run(self.store_num)

            explanation_string = report.explanation['cool_numbers']
            formatted = re.sub('Metric: (.*), Diff: (.*) %, threshold: (.*)%', r"\1 \2", explanation_string)
            formatted_list = re.split(' |\n', formatted)
            self.assertCountEqual(formatted_list, ['mean', '+20.0', 'value_range', '+67.33', 'std', '+66.67', ''])

        with self.subTest('no analyzer should detect shift'):
            self.check = SimpleCheck(mean_threshold=.3, value_range_threshold=.68, std_threshold=.7)
            report = self.check.run(self.store_num)
            self.assertEqual(report.shifted_columns, [])

        with self.subTest('only std should detect shift'):
            self.check = SimpleCheck(mean_threshold=.3, value_range_threshold=.68, std_threshold=.5)
            report = self.check.run(self.store_num)

            explanation_string = report.explanation['cool_numbers']
            formatted = re.sub('Metric: (.*), Diff: (.*) %, threshold: (.*)%', r"\1 \2", explanation_string)
            formatted_list = re.split(' |\n', formatted)
            self.assertCountEqual(formatted_list, ['std', '+67.33', ''])

    def test_run_categorical(self):
        with self.subTest("Test precalculation"):
            report = self.check.run(self.store)

        with self.subTest("Test shifted categorical columns"):
            self.assertEqual(report.shifted_columns, ['shift'])
            self.assertCountEqual(report.examined_columns, ['shift', 'no_shift'])
            self.assertEqual(report.explanation['shift'], "Attribute: 'A' with Diff: +100.0 %, "
                                                          "categorical threshold: +/- 5.0 %\n")

    def test_run_numerical(self):
        with self.subTest("Test precalculation"):
            report = self.check.run(self.store_num)

        with self.subTest('Test shifted numerical columns'):
            self.assertEqual(report.shifted_columns, ['cool_numbers'])
            self.assertCountEqual(report.examined_columns, ['cool_numbers'])

        with self.subTest('Test explanations'):
            explanation_string = report.explanation['cool_numbers']
            formatted = re.sub('Metric: (.*), Diff: (.*) %, threshold: (.*)%', r"\1 \2", explanation_string)
            formatted_list = re.split(' |\n', formatted)
            self.assertCountEqual(formatted_list, ['mean', '+20.0', 'value_range', '+66.67', 'std',
                                                   '+67.33', ''])

    def test_relative_metric_difference(self):
        with self.subTest('Normal case'):
            self.check.data = {'numerical_comparison': {'column': {'metric_name': {'df1': 10.0, 'df2': 5.0}}}}
            self.assertEqual(self.check.relative_metric_difference('column', 'metric_name'), -.5)

            self.check.data = {'numerical_comparison': {'column': {'metric_name': {'df1': 1.2, 'df2': 12.0}}}}
            self.assertEqual(self.check.relative_metric_difference('column', 'metric_name'), 9.0)

        with self.subTest('Both values zero'):
            self.check.data = {'numerical_comparison': {'column': {'metric_name': {'df1': 0, 'df2': 0}}}}
            self.assertEqual(self.check.relative_metric_difference('column', 'metric_name'), 0)

        with self.subTest('Value in df1 zero'):
            self.check.data = {'numerical_comparison': {'column': {'metric_name': {'df1': 0, 'df2': 12.4}}}}
            self.assertEqual(self.check.relative_metric_difference('column', 'metric_name'), 0)

    @mock.patch('shift_detector.checks.simple_check.plt')
    def test_numerical_plots_work(self, mock_plt):
        self.assertFalse(mock_plt.figure.called)

        report = self.check.run(self.store)
        custom_plot_numerical = report.numerical_plot(DataFrame([1, 2, 3]), DataFrame([4, 5, 6]))
        custom_plot_numerical()

        self.assertTrue(mock_plt.figure.called)
        self.assertTrue(mock_plt.figure().add_subplot.called)
        self.assertTrue(mock_plt.show.called)

    @mock.patch('shift_detector.checks.simple_check.plt')
    def test_categorical_plots_work(self, mock_plt):
        self.assertFalse(mock_plt.figure.called)

        report = self.check.run(self.store)
        custom_plot_categorical = report.categorical_plot([([1, 2, 3], [2, 4, 6], ['Heinz', 'Peter', 'Rudolf'],
                                                            'A very important plot')])
        custom_plot_categorical()

        self.assertTrue(mock_plt.figure.called)
        self.assertTrue(mock_plt.figure().add_subplot.called)
        self.assertTrue(mock_plt.show.called)





