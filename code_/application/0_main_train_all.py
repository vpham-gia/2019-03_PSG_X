from copy import copy
from os.path import join, basename, splitext, getsize
from sklearn.decomposition import FastICA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.externals.joblib import dump, load
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline, make_union
from sklearn.tree import DecisionTreeClassifier
from time import time
from tpot.builtins import StackingEstimator
from sklearn.preprocessing import FunctionTransformer
from xgboost import XGBRegressor

import logging

import settings as stg

from code_.domain.data_processing import CategoricalProjectorOnTeamChange, CategoricalProjectorOnAvgDistance as CatProjAvg
from code_.domain.games_info import SeasonFirstHalfAggregator


if __name__ == '__main__':
    stg.enable_logging(log_filename='{}.log'.format(splitext(basename(__file__))[0]),
                       logging_level=logging.DEBUG)

logging.info('Start of script {}'.format(basename(__file__)))
check = input('Have you updated latest TPOT pipelines? [y/n]: ')

if check == 'y':
    logging.info('Step 1 - Player prediction model..')

    logging.info('Step 1 - Load data ..')
    sfha = SeasonFirstHalfAggregator(saved_filename=stg.FILENAME_STATS_AGGREGATED)
    df = sfha.build_players_stats_dataset(sliding_interval_min=5,
                                          list_events_number=stg.EVENTS_COMPUTE_NUMBER,
                                          list_events_with_success_rate=stg.EVENTS_COMPUTE_SUCCESS_RATE,)
    logging.info('Step 1 - .. Done')

    X_player, y_player = df[stg.PLAYER_FEATURES], df[stg.PLAYER_TARGET]

    logging.info('Step 1 - Impute missing values with median ..')
    dump(X_player.median().to_dict(), join(stg.MODELS_DIR, stg.PLAYER_FEATURES_MEDIAN_FILENAME))
    dump(X_player.median().to_dict(), join(stg.SUBMISSION_DIR, stg.PLAYER_FEATURES_MEDIAN_FILENAME))
    X_player.fillna(X_player.median(), inplace=True)
    logging.info('Step 1 - .. Done')

    logging.info('Step 1 - Fit and save pipeline to predict players..')
    player_pipeline = make_pipeline(
        make_union(
            FastICA(tol=0.85),
            FunctionTransformer(copy)
        ),
        ExtraTreesClassifier(n_estimators=75, max_depth=18, bootstrap=False,
                             criterion="gini", max_features=0.1,
                             min_samples_leaf=1, min_samples_split=2)
    )
    player_pipeline_light = make_pipeline(
        make_union(
            FastICA(tol=0.85),
            FunctionTransformer(copy)
        ),
        ExtraTreesClassifier(n_estimators=75, max_depth=17, bootstrap=False,
                             criterion="gini", max_features=0.1,
                             min_samples_leaf=1, min_samples_split=2)
    )
    player_pipeline.fit(X_player, y_player)
    player_pipeline_light.fit(X_player, y_player)
    logging.debug('Step 1 - Fit ok')

    dump(player_pipeline, join(stg.MODELS_DIR, stg.PLAYER_MODEL_NAME), compress=('lz4', 9))
    dump(player_pipeline, join(stg.SUBMISSION_DIR, stg.PLAYER_MODEL_NAME), compress=('lz4', 9))
    dump(player_pipeline_light, join(stg.SUBMISSION_DIR, stg.PLAYER_MODEL_LIGHT_NAME), compress=('lz4', 9))
    file_size = getsize(join(stg.MODELS_DIR, stg.PLAYER_MODEL_NAME))
    logging.debug('Step 1 - Final compression: model size {}'.format(file_size / 1e6))

    load_start = time()
    _ = load(join(stg.MODELS_DIR, stg.PLAYER_MODEL_NAME))
    load_time = time() - load_start
    logging.debug('Step 1 - Load time: {}'.format(load_time))

    logging.info('Step 1 - .. Done')

    logging.info('Step 1 - Done')
    import sys; sys.exit()
    logging.info('--------------------------------')
    logging.info('Step 2 - Next team prediction ..')

    logging.info('Step 2 - Load data ..')
    sfha = SeasonFirstHalfAggregator(saved_filename=stg.FILENAME_NEXT_EVENT)
    df_next_events = sfha.build_next_event_dataset(columns_to_lag=stg.NEXT_EVENT_COLS_TO_LAG,
                                                   lags_to_add=stg.NEXT_EVENT_LAGS)
    logging.info('Step 2 - .. Done')

    logging.info('Step 2 - Feature engineering ..')
    X_team, y_team = df_next_events[stg.NEXT_TEAM_FEATURES], df_next_events[stg.NEXT_TEAM_TARGET]
    next_team_feat_eng_dict = dict()

    for lag in stg.NEXT_EVENT_LAGS:
        cat_proj = CategoricalProjectorOnTeamChange(cat_column_name='{}_lag{}'.format(stg.EVENT_TYPE_COL, lag),
                                                    columns_to_build_change_var=[stg.NEXT_TEAM_TARGET, '{}_lag{}'.format(stg.NEXT_TEAM_TARGET, lag)])
        cat_proj.fit_transform(X_team, y_team)
        dump(cat_proj,
             join(stg.MODELS_DIR, stg.NEXT_TEAM_CAT_PROJ_NAME.format(lag=lag)))

        feat_eng_lag = {'dict': cat_proj.projection_dict, 'mean': cat_proj.mean}
        next_team_feat_eng_dict['lag{}'.format(lag)] = feat_eng_lag

    dump(next_team_feat_eng_dict, join(stg.SUBMISSION_DIR, stg.NEXT_TEAM_FEAT_ENG_DICT))
    logging.info('Step 2 - .. Done')

    logging.info('Step 2 - Fit and save pipeline to predict next team..')
    next_team_pipeline = make_pipeline(
        StackingEstimator(estimator=DecisionTreeClassifier(criterion="entropy", max_depth=4, min_samples_leaf=10, min_samples_split=5)),
        StackingEstimator(estimator=LogisticRegression(C=1.0, dual=False, penalty="l2")),
        StackingEstimator(estimator=DecisionTreeClassifier(criterion="entropy", max_depth=4, min_samples_leaf=17, min_samples_split=5)),
        ExtraTreesClassifier(bootstrap=True, criterion="entropy", max_features=0.7500000000000001, min_samples_leaf=12, min_samples_split=12, n_estimators=100)
    )
    next_team_pipeline.fit(X_team, y_team)
    logging.debug('Step 2 - Fit ok')
    dump(next_team_pipeline, join(stg.MODELS_DIR, stg.NEXT_TEAM_MODEL_NAME), compress=('lz4', 3))
    dump(next_team_pipeline, join(stg.SUBMISSION_DIR, stg.NEXT_TEAM_MODEL_NAME), compress=('lz4', 3))

    file_size = getsize(join(stg.MODELS_DIR, stg.NEXT_TEAM_MODEL_NAME))
    logging.debug('Step 2 - Final compression: model size {}'.format(file_size / 1e6))

    load_start = time()
    _ = load(join(stg.MODELS_DIR, stg.NEXT_TEAM_MODEL_NAME))
    load_time = time() - load_start
    logging.debug('Step 2 - Load time: {}'.format(load_time))
    logging.info('Step 2 - .. Done')

    logging.info('Step 2 - Done')
    logging.info('--------------------------------')
    logging.info('Step 3 - Next coordinates prediction ..')
    logging.info('Step 3 - Data already loaded from Step 2')
    logging.info('Step 3 - Feature engineering ..')
    X_xcoords, y_xcoords = df_next_events[stg.X_PROJ_FEATURES], df_next_events[stg.X_PROJ_TARGET]
    X_ycoords, y_ycoords = df_next_events[stg.Y_PROJ_FEATURES], df_next_events[stg.Y_PROJ_TARGET]

    xcoords_feat_eng_dict, ycoords_feat_eng_dict = dict(), dict()

    for lag in stg.NEXT_EVENT_LAGS:
        cat_proj_xcoords = CatProjAvg(cat_column_name='{}_lag{}'.format(stg.EVENT_TYPE_COL, lag),
                                      columns_to_build_avg_distance=[stg.X_PROJ_TARGET,
                                                                     '{}_lag{}'.format(stg.X_PROJ_TARGET, lag)])
        cat_proj_xcoords.fit_transform(X_xcoords, y_xcoords)
        dump(cat_proj_xcoords, join(stg.MODELS_DIR, stg.X_PROJ_CAT_PROJ_NAME.format(lag=lag)))
        xcoords_feat_eng_lag = {'dict': cat_proj_xcoords.projection_dict,
                                'mean': cat_proj_xcoords.mean}
        xcoords_feat_eng_dict['lag{}'.format(lag)] = xcoords_feat_eng_lag

        cat_proj_ycoords = CatProjAvg(cat_column_name='{}_lag{}'.format(stg.EVENT_TYPE_COL, lag),
                                      columns_to_build_avg_distance=[stg.Y_PROJ_TARGET,
                                                                     '{}_lag{}'.format(stg.Y_PROJ_TARGET, lag)])
        cat_proj_ycoords.fit_transform(X_ycoords, y_ycoords)
        dump(cat_proj_ycoords, join(stg.MODELS_DIR, stg.Y_PROJ_CAT_PROJ_NAME.format(lag=lag)))
        ycoords_feat_eng_lag = {'dict': cat_proj_ycoords.projection_dict,
                                'mean': cat_proj_ycoords.mean}
        ycoords_feat_eng_dict['lag{}'.format(lag)] = ycoords_feat_eng_lag

    dump(xcoords_feat_eng_dict, join(stg.SUBMISSION_DIR, stg.X_PROJ_FEAT_ENG_DICT))
    dump(ycoords_feat_eng_dict, join(stg.SUBMISSION_DIR, stg.Y_PROJ_FEAT_ENG_DICT))
    logging.info('Step 3 - .. Done')

    logging.info('Step 3 - Fit and save pipeline to predict projection of X..')
    xproj_pipeline = make_pipeline(
        MinMaxScaler(),
        XGBRegressor(learning_rate=0.1, max_depth=9, min_child_weight=18, n_estimators=100, nthread=1, subsample=0.9000000000000001)
    )
    xproj_pipeline.fit(X_xcoords, y_xcoords)

    yproj_pipeline = make_pipeline(
        make_union(
            FunctionTransformer(copy),
            FunctionTransformer(copy)
        ),
        XGBRegressor(learning_rate=0.1, max_depth=9, min_child_weight=19, n_estimators=100, nthread=1, subsample=0.9500000000000001)
    )
    yproj_pipeline.fit(X_ycoords, y_ycoords)
    logging.debug('Step 3 - Fit ok')

    dump(xproj_pipeline, join(stg.MODELS_DIR, stg.X_PROJ_MODEL_NAME), compress=('lz4', 3))
    dump(xproj_pipeline, join(stg.SUBMISSION_DIR, stg.X_PROJ_MODEL_NAME), compress=('lz4', 3))
    dump(yproj_pipeline, join(stg.MODELS_DIR, stg.Y_PROJ_MODEL_NAME), compress=('lz4', 3))
    dump(yproj_pipeline, join(stg.SUBMISSION_DIR, stg.Y_PROJ_MODEL_NAME), compress=('lz4', 3))

    file_size_xproj = getsize(join(stg.MODELS_DIR, stg.X_PROJ_MODEL_NAME))
    file_size_yproj = getsize(join(stg.MODELS_DIR, stg.Y_PROJ_MODEL_NAME))
    logging.debug('Step 3 - Final compression: X_proj model size {} | Y_proj model size {}'
                  .format(file_size_xproj / 1e6, file_size_yproj / 1e6))

    load_start = time()
    _ = load(join(stg.MODELS_DIR, stg.X_PROJ_MODEL_NAME))
    _ = load(join(stg.MODELS_DIR, stg.Y_PROJ_MODEL_NAME))
    load_time = time() - load_start
    logging.debug('Step 3 - Load time both (x, y): {}'.format(load_time))

    logging.info('Step 3 - .. Done')
    logging.info('Step 3 - Done')
    logging.info('--------------------------------')
elif check == 'n':
    print('You\'d better update TPOT pipelines before launching this script')
    logging.info('Not latest version of TPOT pipelines - script not run')
else:
    print('Option not understood, program ends')
    logging.info('Incorrect output - script not run')

logging.info('End of script {}'.format(basename(__file__)))
