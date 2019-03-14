from os.path import basename, splitext, join
from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier

import logging
import pandas as pd

from code_.domain.data_processing import DataQualityChecker
from code_.domain.games_info import SeasonFirstHalfAggregator
from code_.domain.predictors import Modeler
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
                                      list_events_with_success_rate=stg.EVENTS_COMPUTE_SUCCESS_RATE,)
logging.info('.. Done')

logging.info('Data quality check ..')
dqc = DataQualityChecker(df=pd.read_csv(join(stg.OUTPUTS_DIR, stg.FILENAME_STATS_AGGREGATED)))
dqc.print_completeness()
dqc.print_min_nb_observations_by_target(target=stg.PLAYER_COL)
logging.info('.. Done')

train, test = train_test_split(df, test_size=0.3, random_state=42)
X_train, y_train = train[stg.PLAYER_FEATURES], train[stg.PLAYER_TARGET]
X_test, y_test = test[stg.PLAYER_FEATURES], test[stg.PLAYER_TARGET]

logging.info('Impute missing values with median ..')
X_test.fillna(X_train.median(), inplace=True)
X_train.fillna(X_train.median(), inplace=True)
logging.info('.. Done')

if stg.BOOL_TPOT_PLAYER:
    logging.info('Starting TPOT - max time: {} min ..'.format(stg.PLAYER_TPOT_LIMIT_TIME))
    pipeline_optimizer = TPOTClassifier(**stg.PLAYER_TPOT_HYPERPARAMS)
    pipeline_optimizer.fit(X_train, y_train)
    logging.info('TPOT Score: {}'.format(pipeline_optimizer.score(X_test, y_test)))
    pipeline_optimizer.export(join(stg.OUTPUTS_DIR, stg.PLAYER_TPOT_FILENAME))
    logging.info('.. Done')
else:
    player_pred = Modeler(model_type=stg.PLAYER_MODEL_TYPE,
                          hyperparameters=stg.PLAYER_MODEL_BASE_HYPERPARAMS,
                          target=stg.PLAYER_TARGET, features=stg.PLAYER_FEATURES)

    if stg.BOOL_TRAIN_PLAYER_MODEL:
        if stg.BOOL_PLAYER_RS:
            logging.info('Cross-validation ..')
            player_pred.perform_random_search_cv(training_data=pd.concat([X_train, y_train], axis=1),
                                                 score='accuracy',
                                                 param_distributions=stg.PLAYER_RANDOM_SEARCH_HYPERPARAMS,
                                                 n_jobs=stg.N_JOBS)
            logging.info('.. Done')

        logging.info('Model to predict players..')
        player_pred.model.set_params(**{'n_jobs': 3})
        player_pred.fit(training_data=pd.concat([X_train, y_train], axis=1))
        logging.debug('Fit ok')
        player_pred.save_model(save_modelname=stg.PLAYER_MODEL_NAME)
        logging.info('.. Done')
    else:
        logging.info('Loading latest model ..')
        player_pred.load_model(save_modelname=stg.PLAYER_MODEL_NAME)
        logging.info('.. Done')

    logging.info('Performance evaluation ..')
    pred = player_pred.predict(test_data=X_test)
    pa = PerformanceAnalyzer(y_true=y_test, y_pred=pred)
    accuracy = pa.compute_classification_accuracy()
    logging.info('Classification accuracy: {}'.format(accuracy))
    logging.info('.. Done')

logging.info('End of script {}'.format(basename(__file__)))
