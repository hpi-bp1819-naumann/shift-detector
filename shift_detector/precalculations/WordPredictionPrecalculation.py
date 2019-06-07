import os
from typing import Tuple

import numpy as np
from gensim.models import FastText
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential

from shift_detector.precalculations.Precalculation import Precalculation
from shift_detector.precalculations.TextEmbeddingPrecalculation import TextEmbeddingPrecalculation


class WordPredictionPrecalculation(Precalculation):

    def __init__(self, ft_window_size=5, ft_size=100, lstm_window=5, num_epochs_predictor=10):
        self.ft_window_size = ft_window_size
        self.ft_size = ft_size
        self.lstm_window = lstm_window
        self.num_epochs_predictor = num_epochs_predictor

        self.verbose = 1

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.ft_window_size == other.ft_window_size \
            and self.ft_size == other.ft_size \
            and self.lstm_window == other.lstm_window

    def __hash__(self):
        return hash((self.ft_window_size, self.ft_size, self.lstm_window))

    def process(self, store) -> dict:
        ft_model = FastText(size=self.ft_size, window=self.ft_window_size, min_count=1, workers=4)
        processed_df1, processed_df2 = store[TextEmbeddingPrecalculation(model=ft_model, agg=None)]

        result = {}
        for column in processed_df1.columns:
            df1_prediction_loss, df2_prediction_loss = self.get_prediction_losses(processed_df1, processed_df2, column)
            result[column] = df1_prediction_loss, df2_prediction_loss

        return result

    def get_prediction_losses(self, processed_df1, processed_df2, column) -> Tuple[float, float]:
        processed_df1 = processed_df1[[column]]
        processed_df2 = processed_df2[[column]]

        processed_df1_train, processed_df1_test = self.split(processed_df1)
        df1_train_x, df1_train_y = self.get_features_and_labels(processed_df1_train)
        df1_test_x, df1_test_y = self.get_features_and_labels(processed_df1_test)

        df2_test_x, df2_test_y = self.get_features_and_labels(processed_df2)

        # build and train model
        prediction_model = self.create_model()
        prediction_model.fit(df1_train_x, df1_train_y,
                             epochs=self.num_epochs_predictor, batch_size=512,
                             callbacks=self.create_callbacks(),
                             verbose=self.verbose,
                             validation_data=(df1_test_x, df1_test_y))

        # get prediction loss for both datasets
        df1_prediction_loss = prediction_model.evaluate(x=df1_test_x, y=df1_test_y)
        df2_prediction_loss = prediction_model.evaluate(x=df2_test_x, y=df2_test_y)

        return df1_prediction_loss, df2_prediction_loss

    def split(self, df, ratio=.8):
        msk = np.random.rand(len(df)) < ratio
        return df[msk], df[~msk]

    def get_features_and_labels(self, df):

        data = [np.array(row[0]) for row in df.values]

        train_data = []

        for row in data:
            if row.shape[0] > self.lstm_window:
                for start_idx in range(row.shape[0] - self.lstm_window - 1):
                    end_idx = start_idx + self.lstm_window + 1
                    train_data += [row[start_idx:end_idx, :]]

        train_data = np.array(train_data)
        print('train_data.shape: ', train_data.shape)

        features = train_data[:, :self.lstm_window, :]
        labels = train_data[:, self.lstm_window, :]

        print(features.shape)
        print(labels.shape)

        return features, labels

    def create_callbacks(self):
        callbacks = []
        callbacks += [ModelCheckpoint('model_checkpoints/model.h5', verbose=self.verbose,
                                      monitor='loss', save_best_only=True, mode='auto')]
        callbacks += [EarlyStopping(monitor='loss', patience=3, verbose=self.verbose,
                                    mode='auto', restore_best_weights=True)]

        if not os.path.exists('model_checkpoints'):
            os.mkdir('model_checkpoints')

        return callbacks

    def create_model(self):
        model = Sequential()
        model.add(LSTM(128, input_shape=(self.lstm_window, self.ft_size)))
        model.add(Dense(self.ft_size))

        model.compile(loss='mean_squared_error', optimizer='adam')
        return model
