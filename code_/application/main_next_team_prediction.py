from os.path import basename, splitext
from sklearn.model_selection import train_test_split

import logging
import pandas as pd

from code_.domain.data_processing import CategoricalProjectorOnTeamChange, DataQualityChecker
from code_.domain.games_info import SeasonFirstHalfAggregator
from code_.domain.predictors import Modeler
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

logging.info('Data quality check - Train set..')
dqc = DataQualityChecker(df=pd.concat([X_train, y_train], axis=1))
dqc.print_completeness()
dqc.print_min_nb_observations_by_target(target=stg.NEXT_TEAM_TARGET)

logging.info('Data quality check - Test set..')
dqc = DataQualityChecker(df=pd.concat([X_test, y_test], axis=1))
dqc.print_completeness()
dqc.print_min_nb_observations_by_target(target=stg.NEXT_TEAM_TARGET)
logging.info('.. Done')

next_team_model = Modeler(model_type=stg.NEXT_TEAM_MODEL_TYPE,
                          hyperparameters=stg.NEXT_TEAM_MODEL_HYPERPARAMS,
                          target=stg.NEXT_TEAM_TARGET,
                          features=stg.NEXT_TEAM_FEATURES)

if stg.BOOL_TRAIN_NEXT_TEAM_MODEL:
    if stg.BOOL_NEXT_TEAM_RS:
        logging.info('Cross-validation ..')
        next_team_model.perform_random_search_cv(training_data=pd.concat([X_train, y_train], axis=1),
                                                 score='accuracy',
                                                 param_distributions=stg.NEXT_TEAM_RANDOM_SEARCH_HYPERPARAMS,
                                                 n_jobs=stg.N_JOBS)
        logging.info('.. Done')

    logging.info('Model to predict next event..')
    next_team_model.model.set_params(**{'n_jobs': 3})
    next_team_model.fit(training_data=pd.concat([X_train, y_train], axis=1))
    logging.debug('Fit ok')
    next_team_model.save_model(save_modelname=stg.NEXT_TEAM_MODEL_NAME)
    logging.info('.. Done')
else:
    logging.info('Loading latest model ..')
    next_team_model.load_model(save_modelname=stg.NEXT_TEAM_MODEL_NAME)
    logging.info('.. Done')

logging.info('Performance evaluation ..')
pred = next_team_model.predict(test_data=X_test)
pa = PerformanceAnalyzer(y_true=y_test, y_pred=pred)
accuracy = pa.compute_classification_accuracy()
logging.info('Classification accuracy: {}'.format(accuracy))
logging.info('.. Done')

logging.info('End of script {}'.format(basename(__file__)))
