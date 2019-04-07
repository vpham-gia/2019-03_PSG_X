from os.path import basename, splitext
from sklearn.model_selection import train_test_split

import logging
import lightgbm as lgb

from code_.domain.data_processing import CategoricalProjectorOnTeamChange
from code_.domain.games_info import SeasonFirstHalfAggregator
from code_.domain.performance_analyzer import PerformanceAnalyzer

import settings as stg

if __name__ == '__main__':
    stg.enable_logging(log_filename='{}.log'.format(splitext(basename(__file__))[0]),
                       logging_level=logging.DEBUG)

logging.info('Start of script {}'.format(basename(__file__)))

logging.info('Load data ..')
sfha = SeasonFirstHalfAggregator(saved_filename=stg.FILENAME_NEXT_EVENT)
df = sfha.build_next_event_dataset(columns_to_lag=stg.NEXT_EVENT_COLS_TO_LAG,
                                   lags_to_add=stg.NEXT_EVENT_LAGS)
logging.info('.. Done')

logging.info('Feature engineering ..')
train, test = train_test_split(df.dropna(), test_size=0.3, random_state=42)
X_train, y_train = train[stg.NEXT_TEAM_FEATURES], train[stg.NEXT_TEAM_TARGET]
X_test, y_test = test[stg.NEXT_TEAM_FEATURES], test[stg.NEXT_TEAM_TARGET]

for lag in stg.NEXT_EVENT_LAGS:
    cat_proj = CategoricalProjectorOnTeamChange(cat_column_name='{}_lag{}'.format(stg.EVENT_TYPE_COL, lag),
                                                columns_to_build_change_var=[stg.NEXT_TEAM_TARGET, '{}_lag{}'.format(stg.NEXT_TEAM_TARGET, lag)])
    cat_proj.fit_transform(X_train, y_train)
    cat_proj.transform(X_test)
logging.info('.. Done')

logging.info('LightGBM')
train_data = lgb.Dataset(X_train, label=y_train)

params = {
    'learning_rate': 0.03,
    'objective': 'binary',
    'metric': 'binary_logloss',
    'n_estimators': 500,
    'num_leaves': 100
}

num_rounds = 500
bst = lgb.train(params, train_data, num_rounds)
logging.info('.. Done')

logging.info('Performance evaluation ..')
pred_proba = bst.predict(X_test)
pred = [1 if x > 0.5 else 0 for x in pred_proba]
pa = PerformanceAnalyzer(y_true=y_test, y_pred=pred)
accuracy = pa.compute_classification_accuracy()
logging.info('Classification accuracy: {}'.format(accuracy))
logging.info('.. Done')

logging.info('End of script {}'.format(basename(__file__)))
