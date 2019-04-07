from os.path import basename, splitext
from sklearn.model_selection import train_test_split

import logging
import lightgbm as lgb

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

player_ids = df[stg.PLAYER_COL].unique().tolist()
mapping_class = range(len(player_ids))
mapping_dict = dict(zip(player_ids, mapping_class))
df[stg.PLAYER_COL] = df[stg.PLAYER_COL].replace(mapping_dict)

train, test = train_test_split(df, test_size=0.3, random_state=42)
X_train, y_train = train[stg.PLAYER_FEATURES], train[stg.PLAYER_TARGET]
X_test, y_test = test[stg.PLAYER_FEATURES], test[stg.PLAYER_TARGET]

logging.info('Impute missing values with median ..')
X_test.fillna(X_train.median(), inplace=True)
X_train.fillna(X_train.median(), inplace=True)
logging.info('.. Done')

logging.info('LightGBM')
train_data = lgb.Dataset(X_train, label=y_train)

params = {
    'learning_rate': 0.01,
    'objective': 'multiclass',
    'num_class': 227,
    'metric': 'multi_logloss',
    # 'max_depth': 20,
    'n_estimators': 200,
    'num_leaves': 31
}

num_rounds = 200
bst = lgb.train(params, train_data, num_rounds)
logging.info('.. Done')

logging.info('Performance evaluation ..')
pred_proba = bst.predict(X_test)
pred = list(map(lambda x: list(x).index(max(x)), pred_proba))
pa = PerformanceAnalyzer(y_true=y_test, y_pred=pred)
accuracy = pa.compute_classification_accuracy()
logging.info('Classification accuracy: {}'.format(accuracy))
logging.info('.. Done')

logging.info('End of script {}'.format(basename(__file__)))
