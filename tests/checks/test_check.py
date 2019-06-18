from unittest import TestCase

from shift_detector.checks.check import Report


class TestReport(TestCase):

    def setUp(self):
        self.examined_columns = ['shift', 'no_shift']
        self.shifted_columns = ['shift']
        self.explanation = {
            'shift': 'shifted'
        }
        self.information = {
            'random information': 'cool information'
        }

        self.report = Report("Check", self.examined_columns, self.shifted_columns, self.explanation, self.information)

    def test_add(self):
        with self.subTest("Test inequality"):
            self.assertRaises(Exception, self.report.__add__, "Not a Report")

        examined_columns = ['shift', 'no_shift', 'another_shift']
        shifted_columns = ['shift', 'another_shift']
        explanation = {
            'shift': 'shifted',
            'another_shift': 'another shift'
        }
        information = {
            'random information': 'better information'
        }

        other_report = Report("", examined_columns, shifted_columns, explanation, information)
        res_report = self.report + other_report

        with self.subTest("Test examined Columns"):
            self.assertCountEqual(['shift', 'no_shift', 'another_shift'], res_report.examined_columns)

        with self.subTest("Test shifted Columns"):
            self.assertCountEqual(['shift', 'another_shift'], res_report.shifted_columns)

    def test_sum_dicts(self):
        another_explanation = {
            'shift': 'shifted',
            'another_shift': 'another shift'
        }
        actual_explanation = Report._Report__sum_dicts(self.report.explanation, another_explanation)

        expected_explanation = {
            'shift': 'shifted\nshifted',
            'another_shift': 'another shift'
        }

        self.assertDictEqual(expected_explanation, actual_explanation)

    def test_str(self):
        print(self.report)
        expected_msg = "Examined Columns: ['shift', 'no_shift']\n" \
                       "Shifted Columns: ['shift']\n" \
                       "\n" \
                       "Column 'shift':\n" \
                       "shifted\n" \
                       "'random information':\n" \
                       "cool information\n"

        self.assertEqual(expected_msg, self.report.__str__())
