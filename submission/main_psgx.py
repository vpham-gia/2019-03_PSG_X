from multiprocessing import Process, Value, Array
from os.path import join
from sklearn.externals.joblib import load
from time import time

import csv
import pandas as pd

from code_.domain.games_info import StatsGameAnalyzer, NextEventInGame

import settings as stg

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def Result(xml_filename='assignment/cleaned_test_set.xml'):
    """Compute result for test set."""
    predicted_player = Value('i', 12)
    predicted_next_event = Array('d', [0, 50, 50])

    p1 = Process(target=predict_player, args=(predicted_player, 'assignment/cleaned_test_set.xml'))
    p2 = Process(target=predict_next_team_and_coords, args=(predicted_next_event, 'assignment/cleaned_test_set.xml'))
    p1.start()
    p2.start()
    p1.join()
    p2.join()

    with open('./res_psgx_vinh_phamgia.csv', mode='w') as result_file:
        prediction_writer = csv.writer(result_file, delimiter=',',
                                       quoting=csv.QUOTE_MINIMAL)
        prediction_writer.writerow([int(predicted_player.value),
                                    int(predicted_next_event[0]),
                                    predicted_next_event[1],
                                    predicted_next_event[2]
                                    ])


def predict_player(player_id, xml_filename='assignment/cleaned_test_set.xml'):
    """Predict player.

    Parameters
    ----------
    player_id: multiprocessing.Value object
        Contains predicted player_id
    xml_filename: string
        Path to access test XML filename

    """
    start = time()
    sga = StatsGameAnalyzer(filename=xml_filename)
    player_stats = sga.build_stats_for_test_set(df_15_min=sga.game,
                                                list_events_number=stg.EVENTS_COMPUTE_NUMBER,
                                                list_events_with_success_rate=stg.EVENTS_COMPUTE_SUCCESS_RATE)

    all_feats = pd.DataFrame(columns=stg.PLAYER_FEATURES)
    player_all_stats = pd.concat([all_feats, player_stats], axis=0, ignore_index=True, sort=False)

    median_train_set = load(join(stg.MODELS_DIR, stg.PLAYER_FEATURES_MEDIAN_FILENAME))
    X_test = player_all_stats.drop(labels=stg.PLAYER_COL, axis=1)\
                             .fillna({col: 0 for col in player_all_stats.columns if col.endswith('_nb')})\
                             .fillna(median_train_set)
    print('Pb 1 - After feature engineering: {}'.format(time() - start))

    player_model = load(join(stg.MODELS_DIR, stg.PLAYER_MODEL_NAME))
    print('Pb 1 - After model load: {}'.format(time() - start))

    player_pred = player_model.predict(X_test)
    print('Pb 1 - After prediction {}'.format(time() - start))

    player_id.value = player_pred[0]


def predict_next_team_and_coords(next_event_array, xml_filename='assignment/cleaned_test_set.xml'):
    """Predict next event characteristics.

    Parameters
    ----------
    next_event_array: multiprocessing.Array object
        Contains predictions for next team, y and x
    xml_filename: string
        Path to access test XML filename

    """
    start = time()
    neig = NextEventInGame(filename=xml_filename)
    df_events = neig.build_next_event_dataset(columns_to_lag=stg.NEXT_EVENT_COLS_TO_LAG,
                                              lags_to_add=stg.NEXT_EVENT_LAGS)

    X_next_event = df_events[stg.NEXT_TEAM_FEATURES].tail(1)
    X_xcoords = df_events[stg.X_PROJ_FEATURES].tail(1)
    X_ycoords = df_events[stg.Y_PROJ_FEATURES].tail(1)

    for lag in stg.NEXT_EVENT_LAGS:
        X_next_event['{}_lag{}'.format(stg.EVENT_TYPE_COL, lag)] = X_next_event['{}_lag{}'.format(stg.EVENT_TYPE_COL, lag)].apply(int)

        cat_proj_team = load(join(stg.MODELS_DIR, stg.NEXT_TEAM_CAT_PROJ_NAME.format(lag=lag)))
        cat_proj_team.transform(X_next_event)

    print('Pb 2 - After feature engineering: {}'.format(time() - start))

    next_team_model = load(join(stg.MODELS_DIR, stg.NEXT_TEAM_MODEL_NAME))
    next_team_pred = next_team_model.predict(X_next_event)[0]
    print('Pb 2 - After model load and prediction: {}'.format(time() - start))

    for lag in stg.NEXT_EVENT_LAGS:
        X_xcoords['{}_lag{}'.format(stg.EVENT_TYPE_COL, lag)] = X_xcoords['{}_lag{}'.format(stg.EVENT_TYPE_COL, lag)].apply(int)

        cat_proj_xcoords = load(join(stg.MODELS_DIR, stg.X_PROJ_CAT_PROJ_NAME.format(lag=lag)))
        cat_proj_xcoords.transform(X_xcoords)

        X_ycoords['{}_lag{}'.format(stg.EVENT_TYPE_COL, lag)] = X_ycoords['{}_lag{}'.format(stg.EVENT_TYPE_COL, lag)].apply(int)

        cat_proj_ycoords = load(join(stg.MODELS_DIR, stg.Y_PROJ_CAT_PROJ_NAME.format(lag=lag)))
        cat_proj_ycoords.transform(X_ycoords)

    print('Pb 3 - After feature engineering: {}'.format(time() - start))

    xcoords_model = load(join(stg.MODELS_DIR, stg.X_PROJ_MODEL_NAME))
    ycoords_model = load(join(stg.MODELS_DIR, stg.Y_PROJ_MODEL_NAME))

    xcoords_along_team1 = xcoords_model.predict(X_xcoords)[0]
    ycoords_along_team1 = ycoords_model.predict(X_ycoords)[0]

    xcoord_pred = next_team_pred * xcoords_along_team1 + (1 - next_team_pred) * (100 - xcoords_along_team1)
    ycoord_pred = next_team_pred * ycoords_along_team1 + (1 - next_team_pred) * (100 - ycoords_along_team1)
    print('Pb 3 - After model load and prediction: {}'.format(time() - start))

    next_event_array[0] = next_team_pred
    next_event_array[1] = ycoord_pred
    next_event_array[2] = xcoord_pred


if __name__ == '__main__':
    # player = predict_player(xml_filename='assignment/cleaned_test_set.xml')
    # next_team, y, x = predict_next_team_and_coords(xml_filename='assignment/cleaned_test_set.xml')
    #
    start = time()
    #
    # player = Value('i', 12)
    # next_event = Array('i', [0, 50, 50])
    #
    # p1 = Process(target=predict_player, args=(player, 'assignment/cleaned_test_set.xml'))
    # p2 = Process(target=predict_next_team_and_coords, args=(next_event, 'assignment/cleaned_test_set.xml'))
    # p1.start()
    # p2.start()
    # p1.join()
    # p2.join()
    #

    Result(xml_filename='assignment/cleaned_test_set.xml')

    print('Time elapsed: {}'.format(time() - start))
