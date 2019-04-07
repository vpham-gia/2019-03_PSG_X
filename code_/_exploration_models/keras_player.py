from os.path import basename, splitext
from sklearn.model_selection import train_test_split

import logging

from code_.domain.games_info import SeasonFirstHalfAggregator
from code_.domain.performance_analyzer import PerformanceAnalyzer

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

import settings as stg

if __name__ == '__main__':
    stg.enable_logging(log_filename='{}.log'.format(splitext(basename(__file__))[0]),
                       logging_level=logging.DEBUG)

seed = 42
np.random.seed(seed)

logging.info('Start of script {}'.format(basename(__file__)))

logging.info('Load data ..')
sfha = SeasonFirstHalfAggregator(saved_filename=stg.FILENAME_STATS_AGGREGATED)
df = sfha.build_players_stats_dataset(sliding_interval_min=5,
                                      list_events_number=stg.EVENTS_COMPUTE_NUMBER,
                                      list_events_with_success_rate=stg.EVENTS_COMPUTE_SUCCESS_RATE)
logging.info('.. Done')

player_ids = df[stg.PLAYER_COL].unique().tolist()
mapping_class = range(len(player_ids))
mapping_player_class = dict(zip(player_ids, mapping_class))
mapping_class_player = dict(zip(mapping_class, player_ids))

train, test = train_test_split(df, test_size=0.3, random_state=42)
train[stg.PLAYER_COL] = train[stg.PLAYER_COL].replace(mapping_player_class)
X_train, y_train = train[stg.PLAYER_FEATURES], train[stg.PLAYER_TARGET]
X_test, y_test = test[stg.PLAYER_FEATURES], test[stg.PLAYER_TARGET]

logging.info('Impute missing values with median ..')
X_test.fillna(X_train.median(), inplace=True)
X_train.fillna(X_train.median(), inplace=True)
logging.info('.. Done')

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y_train)
encoded_train = encoder.transform(y_train)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y_train = np_utils.to_categorical(encoded_train)

logging.info('Keras model')


def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(256, input_dim=len(X_train.columns), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    # model.add(Dense(1024, activation='relu'))
    model.add(Dense(len(df[stg.PLAYER_COL].unique()), activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


keras_model = KerasClassifier(build_fn=baseline_model,
                              epochs=500, batch_size=1024,
                              verbose=1)
keras_model.fit(X_train, dummy_y_train)
logging.info('.. Done')

logging.info('Performance evaluation ..')
pred_classes = keras_model.predict(X_test)
pred_players = list(map(lambda x: mapping_class_player[x], pred_classes))
pa = PerformanceAnalyzer(y_true=y_test, y_pred=pred_players)
accuracy = pa.compute_classification_accuracy()
logging.info('Classification accuracy: {}'.format(accuracy))
logging.info('.. Done')

logging.info('End of script {}'.format(basename(__file__)))
