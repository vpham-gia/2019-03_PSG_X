from os.path import basename, splitext
from sklearn.model_selection import train_test_split
from copy import copy
from sklearn.decomposition import FastICA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer

import logging

from code_.domain.games_info import SeasonFirstHalfAggregator
from code_.domain.performance_analyzer import PerformanceAnalyzer

import settings as stg

if __name__ == '__main__':
    stg.enable_logging(log_filename='{}.log'.format(splitext(basename(__file__))[0]),
                       logging_level=logging.DEBUG)

logging.info('Start of script {}'.format(basename(__file__)))

logging.info('Load data ..')
sfha = SeasonFirstHalfAggregator(saved_filename=stg.FILENAME_STATS_AGGREGATED)
df = sfha.build_players_stats_dataset(sliding_interval_min=5,
                                      list_events_number=stg.EVENTS_COMPUTE_NUMBER,
                                      list_events_with_success_rate=stg.EVENTS_COMPUTE_SUCCESS_RATE)
logging.info('.. Done')

train, test = train_test_split(df, test_size=0.3, random_state=42)
X_train, y_train = train[stg.PLAYER_FEATURES], train[stg.PLAYER_TARGET]
X_test, y_test = test[stg.PLAYER_FEATURES], test[stg.PLAYER_TARGET]

logging.info('Impute missing values with median ..')
X_test.fillna(X_train.median(), inplace=True)
X_train.fillna(X_train.median(), inplace=True)
logging.info('.. Done')

logging.info('Pipeline')
player_pipeline = make_pipeline(
    make_union(
        FastICA(tol=0.85),
        FunctionTransformer(copy)
    ),
    ExtraTreesClassifier(n_estimators=75, max_depth=17, bootstrap=False,
                         criterion="gini", max_features=0.1,
                         min_samples_leaf=1, min_samples_split=2)
)

player_pipeline.fit(X_train, y_train)
logging.info('.. Done')

logging.info('Performance evaluation ..')
pred = player_pipeline.predict(X_test)
pa = PerformanceAnalyzer(y_true=y_test, y_pred=pred)
accuracy = pa.compute_classification_accuracy()
logging.info('Classification accuracy: {}'.format(accuracy))
logging.info('.. Done')

logging.info('End of script {}'.format(basename(__file__)))
