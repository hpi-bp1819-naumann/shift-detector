import re
from unittest import TestCase
import mock

from pandas import DataFrame

from shift_detector.checks.SimpleCheck import SimpleCheck
from shift_detector.precalculations.Store import Store


class TestSimpleCheck(TestCase):

    def setUp(self):
        sales1 = {'shift': ['A'] * 100, 'no_shift': ['C'] * 100}
        sales2 = {'shift': ['B'] * 100, 'no_shift': ['C'] * 100}
        self.df1 = DataFrame.from_dict(sales1)
        self.df2 = DataFrame.from_dict(sales2)
        self.store = Store(self.df1, self.df2)
        self.check = SimpleCheck()

    def test_init(self):
        print("###### \n\n\nParameters missing\n\n\n######")
        self.assertTrue(True)
        # with self.subTest("Test wrong columns"):
        #     # self.assertRaises(TypeError, lambda: DistinctionCheck(['shift', 0]))
        #
        # with self.subTest("Test wrong num epochs"):
        #     # self.assertRaises(TypeError, lambda: DistinctionCheck(num_epochs='wrong'))
        #     # self.assertRaises(ValueError, lambda: DistinctionCheck(num_epochs=0))
        #
        # with self.subTest("Test wrong relative threshold"):
        #     self.assertRaises(TypeError, lambda: DistinctionCheck(relative_threshold='wrong'))
        #     self.assertRaises(ValueError, lambda: DistinctionCheck(relative_threshold=-1))

    def test_run_categorical(self):
        with self.subTest("Test precalculation"):
            report = self.check.run(self.store)

        with self.subTest("Test shifted categorical columns"):
            self.assertEqual(report.shifted_columns, ['shift'])
            self.assertCountEqual(report.examined_columns, ['shift', 'no_shift'])
            self.assertEqual(report.explanation['shift'], 'Attribute: A with Diff: 1.0\n')

    def test_run_numerical(self):
        with self.subTest("Test precalculation"):
            numbers1 = {'cool_numbers': [1, 2, 3, 4]}
            numbers2 = {'cool_numbers': [1, 2, 3, 6]}
            self.df1 = DataFrame.from_dict(numbers1)
            self.df2 = DataFrame.from_dict(numbers2)
            self.store = Store(self.df1, self.df2)
            self.check = SimpleCheck()
            report = self.check.run(self.store)

        with self.subTest('Test shifted numerical columns'):
            self.assertEqual(report.shifted_columns, ['cool_numbers'])
            self.assertCountEqual(report.examined_columns, ['cool_numbers'])

        with self.subTest('Test explanations'):
            explanation_string = report.explanation['cool_numbers']
            print(explanation_string, '\n')
            formatted = explanation_string.replace('Metric: ', '').replace('with Diff: ', '').replace(' %', '')
            formatted_list = re.split(' |\n', formatted)
            self.assertCountEqual(formatted_list, ['mean', '+20.0', 'max', '+50.0', 'quartile_3', '+15.38', 'std',
                                                   '+67.33', ''])

    def test_relative_metric_difference(self):
        with self.subTest('Normal case'):
            self.check.data = {'numerical_comparison': {'column': {'metric_name': {'df1': 10.0, 'df2': 5.0}}}}
            self.assertEqual(self.check.relative_metric_difference('column', 'metric_name'), -50)

            self.check.data = {'numerical_comparison': {'column': {'metric_name': {'df1': 1.2, 'df2': 12.0}}}}
            self.assertEqual(self.check.relative_metric_difference('column', 'metric_name'), 900)

        with self.subTest('Both values zero'):
            self.check.data = {'numerical_comparison': {'column': {'metric_name': {'df1': 0, 'df2': 0}}}}
            self.assertEqual(self.check.relative_metric_difference('column', 'metric_name'), 0)

        with self.subTest('Value in df1 zero'):
            self.check.data = {'numerical_comparison': {'column': {'metric_name': {'df1': 0, 'df2': 12.4}}}}
            self.assertEqual(self.check.relative_metric_difference('column', 'metric_name'), 0)


    @mock.patch('shift_detector.checks.SimpleCheck.plt')
    def test_numerical_plots_work(self, mock_plt):
        self.assertFalse(mock_plt.figure.called)

        report = self.check.run(self.store)
        custom_plot_numerical = report.numerical_plot(DataFrame([1, 2, 3]), DataFrame([4, 5, 6]))
        custom_plot_numerical()

        self.assertTrue(mock_plt.figure.called)
        self.assertTrue(mock_plt.figure().add_subplot.called)
        self.assertTrue(mock_plt.show.called)

    @mock.patch('shift_detector.checks.SimpleCheck.plt')
    def test_categorical_plots_work(self, mock_plt):
        self.assertFalse(mock_plt.figure.called)

        report = self.check.run(self.store)
        custom_plot_categorical = report.categorical_plot([([1, 2, 3], [2, 4, 6], ['Heinz', 'Peter', 'Rudolf'],
                                                            'A very important plot')])
        custom_plot_categorical()

        self.assertTrue(mock_plt.figure.called)
        self.assertTrue(mock_plt.figure().add_subplot.called)
        self.assertTrue(mock_plt.figure().show.called)





