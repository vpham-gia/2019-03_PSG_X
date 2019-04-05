from os.path import basename, splitext, join, getsize
from sklearn.model_selection import train_test_split
from copy import copy
from sklearn.decomposition import FastICA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.externals.joblib import dump, load
from sklearn.pipeline import make_pipeline, make_union
from time import time
from sklearn.preprocessing import FunctionTransformer

import logging
import pandas as pd

from code_.domain.games_info import SeasonFirstHalfAggregator
from code_.domain.performance_analyzer import PerformanceAnalyzer

import settings as stg

if __name__ == '__main__':
    stg.enable_logging(log_filename='{}.log'.format(splitext(basename(__file__))[0]),
                       logging_level=logging.INFO)

logging.info('Load data ..')
sfha = SeasonFirstHalfAggregator(saved_filename=stg.FILENAME_STATS_AGGREGATED)
df = sfha.build_players_stats_dataset(sliding_interval_min=5,
                                      list_events_number=stg.EVENTS_COMPUTE_NUMBER,
                                      list_events_with_success_rate=stg.EVENTS_COMPUTE_SUCCESS_RATE)
train, test = train_test_split(df, test_size=0.3, random_state=42)
logging.info('.. Done')

df_nb_trees_characteristics = pd.DataFrame(columns=['ntree', 'file_size_mo', 'load_time', 'accuracy_test_set', 'compression'])
for nb in range(50, 101, 5):
    for depth in range(15, 21, 1):
        logging.info('Pipeline - ntree {}, depth {} ..'.format(nb, depth))

        X_train, y_train = train[stg.PLAYER_FEATURES], train[stg.PLAYER_TARGET]
        X_test, y_test = test[stg.PLAYER_FEATURES], test[stg.PLAYER_TARGET]

        logging.debug('.. Impute missing values with median ..')
        X_test.fillna(X_train.median(), inplace=True)
        X_train.fillna(X_train.median(), inplace=True)
        logging.debug('.. .. Done')

        player_pipeline = make_pipeline(
            make_union(
                FastICA(tol=0.4),
                FunctionTransformer(copy)
            ),
            ExtraTreesClassifier(n_estimators=nb, max_depth=depth,
                                 bootstrap=False, criterion="gini", max_features=0.1,
                                 min_samples_leaf=1, min_samples_split=2)
        )

        player_pipeline.fit(X_train, y_train)
        logging.debug('.. Fit ok')

        pred_pipeline = player_pipeline.predict(X_test)
        pa_pipeline = PerformanceAnalyzer(y_true=y_test, y_pred=pred_pipeline)
        acc_pipeline = pa_pipeline.compute_classification_accuracy()

        for compression in range(9, 10):
            dump(player_pipeline, join(stg.MODELS_DIR, 'tmp_player_pipeline.joblib'),
                 compress=('lz4', compression))
            logging.debug('.. Dump ok')

            load_start = time()
            _ = load(join(stg.MODELS_DIR, 'tmp_player_pipeline.joblib'))
            load_time = time() - load_start

            df_nb = pd.DataFrame({'ntree': nb, 'max_depth': depth,
                                  'file_size_mo': getsize(join(stg.MODELS_DIR, 'tmp_player_pipeline.joblib')) / 1000000,
                                  'load_time': load_time,
                                  'accuracy_test_set': acc_pipeline,
                                  'compression': compression},
                                 index=[0])

            df_nb_trees_characteristics = pd.concat(objs=[df_nb_trees_characteristics, df_nb],
                                                    axis=0, ignore_index=True, sort=False)
            df_nb_trees_characteristics.to_csv(join(stg.OUTPUTS_DIR, '2019-04-05_etc_tests_for_constraints_fastica04.csv'),
                                               index=False)
        logging.info('.. Done')

logging.info('End of script {}'.format(basename(__file__)))
