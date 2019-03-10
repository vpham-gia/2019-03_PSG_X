from os.path import join
from sklearn.model_selection import train_test_split

import logging
import pandas as pd

from code_.domain.data_processing import DataQualityChecker
from code_.domain.games_stats import SeasonFirstHalfAggregator
from code_.domain.predictors import PlayerPredictor
from code_.domain.performance_analyzer import PerformanceAnalyzer

import settings as stg

if __name__ == '__main__':
    stg.enable_logging(log_filename='app.log', logging_level=logging.INFO)

logging.info('Start of script {}'.format(__file__))

logging.info('Load data ..')
sfha = SeasonFirstHalfAggregator(sliding_interval_min=5,
                                 list_events_number=['4', '7', '10', '16', '17', '18', '19', '44', '61'],
                                 list_events_with_success_rate=['1'],
                                 saved_filename=stg.FILENAME_STATS_AGGREGATED)
df = sfha.build_dataset()
logging.info('.. Done')

logging.info('Data quality check ..')
dqc = DataQualityChecker(filename=stg.FILENAME_STATS_AGGREGATED)
dqc.print_completeness()
dqc.print_min_nb_observations_by_target(target=stg.PLAYER_COL)
logging.info('.. Done')

logging.info('Model to predict players..')
train, test = train_test_split(df.dropna(), test_size=0.3, random_state=42)

player_pred = PlayerPredictor(model_type=stg.PLAYER_MODEL_TYPE,
                              hyperparameters=stg.PLAYER_MODEL_HYPERPARAMS,
                              target=stg.PLAYER_TARGET,
                              features=stg.PLAYER_FEATURES)
player_pred.fit(training_data=train)
player_pred.save_model()
logging.info('.. Done')

logging.info('Performance evaluation ..')
pred = player_pred.predict(test_data=test[stg.PLAYER_FEATURES])
pa = PerformanceAnalyzer(y_true=test[stg.PLAYER_COL].values, y_pred=pred)
accuracy = pa.compute_classification_accuracy()
logging.info('Classification accuracy: {}'.format(accuracy))
logging.info('.. Done')

logging.info('End of script {}'.format(__file__))
