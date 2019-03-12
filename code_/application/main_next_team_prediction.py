from os.path import basename, splitext
from sklearn.model_selection import train_test_split

import logging
import pandas as pd

from code_.domain.data_processing import CategoricalProjector, DataQualityChecker
from code_.domain.games_info import SeasonFirstHalfAggregator
from code_.domain.predictors import PlayerPredictor
from code_.domain.performance_analyzer import PerformanceAnalyzer

import settings as stg

if __name__ == '__main__':
    stg.enable_logging(log_filename='{}.log'.format(splitext(basename(__file__))[0]),
                       logging_level=logging.DEBUG)

logging.info('Start of script {}'.format(basename(__file__)))

logging.info('Load data ..')
sfha = SeasonFirstHalfAggregator(sliding_interval_min=5,
                                 list_events_number=stg.EVENTS_COMPUTE_NUMBER,
                                 list_events_with_success_rate=stg.EVENTS_COMPUTE_SUCCESS_RATE,
                                 saved_filename=stg.FILENAME_STATS_AGGREGATED)
df = sfha.build_next_team_dataset(columns_to_lag=stg.NEXT_TEAM_COLS_TO_LAG)
logging.info('.. Done')

logging.info('Data quality check ..')
logging.info('.. Done')

logging.info('Feature engineering ..')
train, test = train_test_split(df.dropna(), test_size=0.3, random_state=42)
X_train, y_train = train[stg.NEXT_TEAM_FEATURES], train[stg.NEXT_TEAM_TARGET]
X_test, y_test = test[stg.NEXT_TEAM_FEATURES], test[stg.NEXT_TEAM_TARGET]

cat_proj = CategoricalProjector(column_to_substitute='{}_lag1'.format(stg.EVENT_TYPE_COL),
                                columns_to_build_change_var=[stg.NEXT_TEAM_TARGET, '{}_lag1'.format(stg.NEXT_TEAM_TARGET)])
cat_proj.fit_transform(X_train, y_train)
cat_proj.transform(X_test)
logging.info('.. Done')

logging.info('Data quality check - Train set..')
dqc = DataQualityChecker(df=pd.concat([X_train, y_train], axis=1))
dqc.print_completeness()
dqc.print_min_nb_observations_by_target(target=stg.TEAM_COL)

logging.info('Data quality check - Test set..')
dqc = DataQualityChecker(df=pd.concat([X_test, y_test], axis=1))
dqc.print_completeness()
dqc.print_min_nb_observations_by_target(target=stg.TEAM_COL)
logging.info('.. Done')

# TODO: refactor name to have generic predictor
next_team_model = PlayerPredictor(model_type=stg.NEXT_TEAM_MODEL_TYPE,
                                  hyperparameters=stg.NEXT_TEAM_MODEL_HYPERPARAMS,
                                  target=stg.NEXT_TEAM_TARGET,
                                  features=stg.NEXT_TEAM_FEATURES)

if stg.BOOL_NEXT_TEAM_RS:
    logging.info('Cross-validation ..')
    next_team_model.perform_random_search_cv(training_data=pd.concat([X_train, y_train], axis=1),
                                             score='accuracy',
                                             param_distributions=stg.NEXT_TEAM_RANDOM_SEARCH_HYPERPARAMS)
    logging.info('.. Done')

logging.info('Model to predict players..')
next_team_model.model.set_params(**{'n_jobs': 3})
next_team_model.fit(training_data=pd.concat([X_train, y_train], axis=1))
next_team_model.save_model(save_modelname=stg.NEXT_TEAM_MODEL_NAME)
logging.info('.. Done')

logging.info('Performance evaluation ..')
pred = next_team_model.predict(test_data=X_test)
pa = PerformanceAnalyzer(y_true=y_test, y_pred=pred)
accuracy = pa.compute_classification_accuracy()
logging.info('Classification accuracy: {}'.format(accuracy))
logging.info('.. Done')

logging.info('End of script {}'.format(basename(__file__)))
