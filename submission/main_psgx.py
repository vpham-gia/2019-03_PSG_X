from multiprocessing import Process, Value, Array
from os.path import splitext, basename
from psutil import virtual_memory
from sklearn.externals.joblib import load

import csv
import logging
import os
import pandas as pd
import sys
import time

from games_info import StatsGameAnalyzer, NextEventInGame

import settings as stg

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

sys.path.append(os.getcwd())

if __name__ == '__main__':
    stg.enable_logging(log_filename='{}.log'.format(splitext(basename(__file__))[0]),
                       logging_level=logging.INFO)


def Result(xml_filename='instructions/cleaned_test_set.xml'):
    """Compute result for test set."""
    predicted_player = Value('i', 59957)
    predicted_next_event = Array('d', [0, 50, 50])

    p1 = Process(target=predict_player, args=(predicted_player, 'instructions/cleaned_test_set.xml'))
    p2 = Process(target=predict_next_team_and_coords, args=(predicted_next_event, 'instructions/cleaned_test_set.xml'))
    p1.start()
    p2.start()
    p1.join()
    p2.join()

    with open('./res_psgx.csv', mode='w') as result_file:
        prediction_writer = csv.writer(result_file, delimiter=',',
                                       quoting=csv.QUOTE_MINIMAL)
        prediction_writer.writerow([int(predicted_player.value),
                                    int(predicted_next_event[0]),
                                    predicted_next_event[1],
                                    predicted_next_event[2]
                                    ])


def predict_player(player_id, xml_filename='instructions/cleaned_test_set.xml'):
    """Predict player.

    Parameters
    ----------
    player_id: multiprocessing.Value object
        Contains predicted player_id
    xml_filename: string
        Path to access test XML filename

    """
    start = time.time()
    sga = StatsGameAnalyzer(filename=xml_filename)
    player_stats = sga.build_stats_for_test_set(df_15_min=sga.game.head(-10),
                                                list_events_number=stg.EVENTS_COMPUTE_NUMBER,
                                                list_events_with_success_rate=stg.EVENTS_COMPUTE_SUCCESS_RATE)

    all_feats = pd.DataFrame(columns=stg.PLAYER_FEATURES)
    player_all_stats = pd.concat([all_feats, player_stats], axis=0, ignore_index=True, sort=False)

    median_train_set = load(stg.PLAYER_FEATURES_MEDIAN_FILENAME)
    X_test = player_all_stats.drop(labels=stg.PLAYER_COL, axis=1)\
                             .fillna({col: 0 for col in player_all_stats.columns if col.endswith('_nb')})\
                             .fillna(median_train_set)
    logging.info('Pb 1 - After feature engineering: {}'.format(time.time() - start))

    ram_go = virtual_memory().total / 1e9
    if ram_go >= 5:
        player_model = load(stg.PLAYER_MODEL_NAME)
    else:
        logging.info('.. RAM memory lower than 5 Go - Light model loaded')
        player_model = load(stg.PLAYER_MODEL_LIGHT_NAME)
    logging.info('Pb 1 - After model load: {}'.format(time.time() - start))

    player_pred = player_model.predict(X_test)
    logging.info('Pb 1 - After prediction {}'.format(time.time() - start))

    player_id.value = player_pred[0]


def predict_next_team_and_coords(next_event_array, xml_filename='instructions/cleaned_test_set.xml'):
    """Predict next event characteristics.

    Parameters
    ----------
    next_event_array: multiprocessing.Array object
        Contains predictions for next team, y and x
    xml_filename: string
        Path to access test XML filename

    """
    start = time.time()
    neig = NextEventInGame(filename=xml_filename)
    df_events = neig.build_next_event_dataset(columns_to_lag=stg.NEXT_EVENT_COLS_TO_LAG,
                                              lags_to_add=stg.NEXT_EVENT_LAGS)

    X_next_event = df_events[stg.NEXT_TEAM_FEATURES].tail(1)
    X_xcoords = df_events[stg.X_PROJ_FEATURES].tail(1)
    X_ycoords = df_events[stg.Y_PROJ_FEATURES].tail(1)

    next_team_feat_eng_dict = load(stg.NEXT_TEAM_FEAT_ENG_DICT)
    xcoords_feat_eng_dict = load(stg.X_PROJ_FEAT_ENG_DICT)
    ycoords_feat_eng_dict = load(stg.Y_PROJ_FEAT_ENG_DICT)

    for lag in stg.NEXT_EVENT_LAGS:
        LAG_COLUMN = '{}_lag{}'.format(stg.EVENT_TYPE_COL, lag)
        LAG_FIRST_KEY = 'lag{}'.format(lag)

        X_next_event[LAG_COLUMN] = X_next_event[LAG_COLUMN].apply(int)\
                                                           .map(next_team_feat_eng_dict[LAG_FIRST_KEY]['dict'])\
                                                           .fillna(value=next_team_feat_eng_dict[LAG_FIRST_KEY]['mean'])

    logging.info('Pb 2 - After feature engineering: {}'.format(time.time() - start))

    next_team_model = load(stg.NEXT_TEAM_MODEL_NAME)
    next_team_pred = next_team_model.predict(X_next_event)[0]
    logging.info('Pb 2 - After model load and prediction: {}'.format(time.time() - start))

    for lag in stg.NEXT_EVENT_LAGS:
        LAG_COLUMN = '{}_lag{}'.format(stg.EVENT_TYPE_COL, lag)
        LAG_FIRST_KEY = 'lag{}'.format(lag)

        X_xcoords[LAG_COLUMN] = X_xcoords[LAG_COLUMN].apply(int)\
                                                     .map(xcoords_feat_eng_dict[LAG_FIRST_KEY]['dict'])\
                                                     .fillna(value=xcoords_feat_eng_dict[LAG_FIRST_KEY]['mean'])

        X_ycoords[LAG_COLUMN] = X_ycoords[LAG_COLUMN].apply(int)\
                                                     .map(ycoords_feat_eng_dict[LAG_FIRST_KEY]['dict'])\
                                                     .fillna(value=ycoords_feat_eng_dict[LAG_FIRST_KEY]['mean'])

    logging.info('Pb 3 - After feature engineering: {}'.format(time.time() - start))

    xcoords_model = load(stg.X_PROJ_MODEL_NAME)
    ycoords_model = load(stg.Y_PROJ_MODEL_NAME)

    xcoords_along_team1 = xcoords_model.predict(X_xcoords)[0]
    ycoords_along_team1 = ycoords_model.predict(X_ycoords)[0]

    xcoord_pred = next_team_pred * xcoords_along_team1 + (1 - next_team_pred) * (100 - xcoords_along_team1)
    ycoord_pred = next_team_pred * ycoords_along_team1 + (1 - next_team_pred) * (100 - ycoords_along_team1)
    logging.info('Pb 3 - After model load and prediction: {}'.format(time.time() - start))

    next_event_array[0] = next_team_pred
    next_event_array[1] = ycoord_pred
    next_event_array[2] = xcoord_pred


if __name__ == '__main__':
    start = time.time()
    Result(xml_filename='cleaned_test_set.xml')
    logging.info('Time elapsed: {}'.format(time.time() - start))
