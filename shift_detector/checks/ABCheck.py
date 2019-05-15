from datawig import SimpleImputer
from datawig.iterators import ImputerIterDf
from datawig.utils import random_split
from datawig.utils import logger as dataWig_logger
import pandas as pd
from typing import Tuple
from sklearn.metrics import classification_report
import numpy as np
import logging as logger
from shift_detector.checks.Check import Check, Report
from shift_detector.Detector import ColumnType
from shift_detector.preprocessors.Default import Default
from typing import Dict

class ABReport(Report):

    def get_columns_with_shift(self):
        return self.check_result['relevant_columns']

    def get_classification_report(self):
        return self.check_result['classification_report']

    def get_f1_score(self, className: str):
        return self.check_result['classification_report'][className]["f1-score"]

    def print_report(self):
        y_true = self.check_result['y_true']
        y_pred = self.check_result['y_pred']
        report = classification_report(y_true, y_pred)
        classification = classification_report(y_true, y_pred, output_dict=True)
        print("Classification Report:")
        print(report)
        print("F1 score for df1:", classification["A"]["f1-score"])
        print("F1 score for df2:", classification["B"]["f1-score"])
        print("Relevant Columns:", self.get_columns_with_shift())

class ABCheck(Check):

    def __init__(self, columns):
        super().__init__()

        self.columns = columns
        self.output_column = '__shift_detector__dataset'
        self.output_path = 'tmp/basicChecks_params'

        self.train_df = None
        self.test_df = None

        self.imputer = None

    @staticmethod
    def name() -> str:
        return "AB Check"

    @staticmethod
    def report_class():
        return ABReport

    def needed_preprocessing(self) -> Dict:
        return {
            ColumnType.numeric: Default(),
            ColumnType.categorical: Default(),
            ColumnType.text: Default()
        }   

    def run(self) -> Dict:
        """
        Runs check on provided columns
        :return: result of the check
        """
        self.imputer = SimpleImputer(
            input_columns=self.columns,
            output_column=self.output_column,
            output_path=self.output_path)

        df1 = pd.concat([df1 for (df1, df2) in self.data.values()], axis=1).dropna()
        df2 = pd.concat([df2 for (df1, df2) in self.data.values()], axis=1).dropna()

        self.train_df, self.test_df = self.prepare_datasets(df1, df2)
        self.imputer.fit(self.train_df, self.test_df)

        imputed = self.imputer.predict(self.test_df)
        y_true, y_pred = imputed[self.output_column], imputed[self.output_column + '_imputed']

        return {
            'y_true': y_true,
            'y_pred': y_pred,
            'relevant_columns': self.relevant_columns(self.columns)
        }

    def label_datasets(self, first_df: pd.DataFrame, second_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Set labels of the first dataframe to 'A' and those of the second dataframe to 'B'
        :param first_df: first dataset 
        :param second_df: second dataset
        :return: tuple of labeled datasets
        """
        first_df[self.output_column] = 'A'
        second_df[self.output_column] = 'B'

        return first_df, second_df

    def sample_datasets(self, first_df: pd.DataFrame, second_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Sample datasets to length of shorter dataset
        :param first_df: first dataset 
        :param second_df: second dataset
        :return: tuple of sampled datasets
        """
        min_len = min(len(first_df), len(second_df))
        return first_df.sample(n=min_len), second_df.sample(n=min_len)

    def prepare_datasets(self, first_df: pd.DataFrame, second_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create a train and a test dataset, in which the number number of tuples 
        that come from the first and the number of those from the second dataset are equal
        :param first_df: first dataset 
        :param second_df: second dataset
        :return: tuple of train and test dataset
        """
        first_df, second_df = self.label_datasets(first_df, second_df)
        first_df_sampled, second_df_sampled = self.sample_datasets(first_df, second_df)
        
        first_df_train, first_df_test = random_split(first_df_sampled)
        second_df_train, second_df_test = random_split(second_df_sampled)
        
        train_df = pd.concat([first_df_train, second_df_train])
        test_df = pd.concat([first_df_test, second_df_test])

        return train_df, test_df

    def get_loss(self, df):
        iterator = ImputerIterDf(
            data_frame=df,
            data_columns=self.imputer.imputer.data_encoders,
            label_columns=self.imputer.imputer.label_encoders,
            batch_size=self.imputer.imputer.batch_size
        )
        return self.imputer.imputer.module.predict(iterator)[0].asnumpy().mean()

    def loss_with_random_permutation(self, df, column):
        losses = []
        for _ in range(5):
            df_permutation = df.copy().sort_values(self.output_column, ascending=True)
            mid_row = int(df_permutation.size / 2)

            first_rand_permutation = np.random.permutation(df_permutation[column][:mid_row].values)
            second_rand_permutation = np.random.permutation(df_permutation[column][mid_row:].values)

            df_permutation[column][:len(second_rand_permutation)] = second_rand_permutation
            df_permutation[column][len(second_rand_permutation):] = first_rand_permutation

            losses += [self.get_loss(df_permutation)]
        return np.asarray(losses).mean()

    def relevant_columns(self, columns, thresh_percentage=.01):
        base_loss = self.get_loss(self.test_df)
        relevant_columns = []

        for col in columns:
            loss = self.loss_with_random_permutation(self.test_df, col)
            if loss > base_loss * (1 + thresh_percentage):
                relevant_columns += [col]

        return relevant_columns
