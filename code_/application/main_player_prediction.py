from os.path import basename, splitext, join
from sklearn.model_selection import train_test_split

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

train, test = train_test_split(df.dropna(), test_size=0.3, random_state=42)

player_pred = Modeler(model_type=stg.PLAYER_MODEL_TYPE,
                      hyperparameters=stg.PLAYER_MODEL_HYPERPARAMS,
                      target=stg.PLAYER_TARGET, features=stg.PLAYER_FEATURES)

if stg.BOOL_TRAIN_PLAYER_MODEL:
    if stg.BOOL_PLAYER_RS:
        logging.info('Cross-validation ..')
        player_pred.perform_random_search_cv(training_data=train, score='accuracy',
                                             param_distributions=stg.PLAYER_RANDOM_SEARCH_HYPERPARAMS,
                                             n_jobs=stg.N_JOBS)
        logging.info('.. Done')

    logging.info('Model to predict players..')
    player_pred.model.set_params(**{'n_jobs': 3})
    player_pred.fit(training_data=train)
    logging.debug('Fit ok')
    player_pred.save_model(save_modelname=stg.PLAYER_MODEL_NAME)
    logging.info('.. Done')
else:
    logging.info('Loading latest model ..')
    player_pred.load_model(save_modelname=stg.PLAYER_MODEL_NAME)
    logging.info('.. Done')

logging.info('Performance evaluation ..')
pred = player_pred.predict(test_data=test[stg.PLAYER_FEATURES])
pa = PerformanceAnalyzer(y_true=test[stg.PLAYER_COL].values, y_pred=pred)
accuracy = pa.compute_classification_accuracy()
logging.info('Classification accuracy: {}'.format(accuracy))
logging.info('.. Done')

logging.info('End of script {}'.format(basename(__file__)))
