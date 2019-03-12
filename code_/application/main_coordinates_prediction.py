from os.path import basename, splitext
from sklearn.model_selection import train_test_split

import logging
import pandas as pd

from code_.domain.data_processing import CategoricalProjector, DataQualityChecker
from code_.domain.games_info import SeasonFirstHalfAggregator
from code_.domain.predictors import Classificator
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
X_train, y_train = train[stg.COORDS_FEATURES], train[stg.COORDS_TARGET]
X_test, y_test = test[stg.COORDS_FEATURES], test[stg.COORDS_TARGET]

# for lag in stg.NEXT_EVENT_LAGS:
#     cat_proj = CategoricalProjector(column_to_substitute='{}_lag{}'.format(stg.EVENT_TYPE_COL, lag),
#                                     columns_to_build_change_var=[stg.COORDS_TARGET, '{}_lag{}'.format(stg.COORDS_TARGET, lag)])
#     cat_proj.fit_transform(X_train, y_train)
#     cat_proj.transform(X_test)
# logging.info('.. Done')

logging.info('Data quality check - Train set..')
dqc = DataQualityChecker(df=pd.concat([X_train, y_train], axis=1))
dqc.print_completeness()
dqc.print_min_nb_observations_by_target(target=stg.COORDS_TARGET)

logging.info('Data quality check - Test set..')
dqc = DataQualityChecker(df=pd.concat([X_test, y_test], axis=1))
dqc.print_completeness()
dqc.print_min_nb_observations_by_target(target=stg.COORDS_TARGET)
logging.info('.. Done')

coords_model = Classificator(model_type=stg.COORDS_MODEL_TYPE,
                             hyperparameters=stg.COORDS_MODEL_HYPERPARAMS,
                             target=stg.COORDS_TARGET,
                             features=stg.COORDS_FEATURES)

if stg.BOOL_TRAIN_COORDS_MODEL:
    if stg.BOOL_COORDS_RS:
        logging.info('Cross-validation ..')
        coords_model.perform_random_search_cv(training_data=pd.concat([X_train, y_train], axis=1),
                                              score='accuracy',
                                              param_distributions=stg.COORDS_RANDOM_SEARCH_HYPERPARAMS)
        logging.info('.. Done')

    logging.info('Model to predict coordinates..')
    coords_model.model.set_params(**{'n_jobs': 3})
    coords_model.fit(training_data=pd.concat([X_train, y_train], axis=1))
    logging.debug('Fit ok')
    coords_model.save_model(save_modelname=stg.COORDS_MODEL_NAME)
    logging.info('.. Done')
else:
    logging.info('Loading latest model ..')
    coords_model.load_model(save_modelname=stg.COORDS_MODEL_NAME)
    logging.info('.. Done')

logging.info('Performance evaluation ..')
pred = coords_model.predict(test_data=X_test)
pa = PerformanceAnalyzer(y_true=y_test, y_pred=pred)
accuracy = pa.compute_accuracy_l2_error()
logging.info('L2 accuracy: {}'.format(accuracy))
logging.info('.. Done')

logging.info('End of script {}'.format(basename(__file__)))
