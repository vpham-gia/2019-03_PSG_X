from os.path import basename, splitext
from sklearn.model_selection import train_test_split

import logging
import pandas as pd

from code_.domain.data_processing import CategoricalProjectorOnAvgDistance as CatProjAvg, DataQualityChecker
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

X_train_xcoords, y_train_xcoords = train[stg.X_PROJ_FEATURES], train[stg.X_PROJ_TARGET]
X_test_xcoords, y_test_xcoords = test[stg.X_PROJ_FEATURES], test[stg.X_PROJ_TARGET]

X_train_ycoords, y_train_ycoords = train[stg.Y_PROJ_FEATURES], train[stg.Y_PROJ_TARGET]
X_test_ycoords, y_test_ycoords = test[stg.Y_PROJ_FEATURES], test[stg.Y_PROJ_TARGET]

for lag in stg.NEXT_EVENT_LAGS:
    cat_proj_xcoords = CatProjAvg(cat_column_name='{}_lag{}'.format(stg.EVENT_TYPE_COL, lag),
                                  columns_to_build_avg_distance=[stg.X_PROJ_TARGET,
                                                                 '{}_lag{}'.format(stg.X_PROJ_TARGET, lag)])
    cat_proj_xcoords.fit_transform(X_train_xcoords, y_train_xcoords)
    cat_proj_xcoords.transform(X_test_xcoords)

    cat_proj_ycoords = CatProjAvg(cat_column_name='{}_lag{}'.format(stg.EVENT_TYPE_COL, lag),
                                  columns_to_build_avg_distance=[stg.Y_PROJ_TARGET,
                                                                 '{}_lag{}'.format(stg.Y_PROJ_TARGET, lag)])
    cat_proj_ycoords.fit_transform(X_train_ycoords, y_train_ycoords)
    cat_proj_ycoords.transform(X_test_ycoords)

logging.info('.. Done')

logging.info('Data quality check - Train set X coordinate ..')
dqc = DataQualityChecker(df=pd.concat([X_train_xcoords, y_train_xcoords], axis=1))
dqc.print_completeness()

logging.info('Data quality check - Test set X coordinate ..')
dqc = DataQualityChecker(df=pd.concat([X_test_xcoords, y_test_xcoords], axis=1))
dqc.print_completeness()
logging.info('.. Done')

logging.info('Data quality check - Train set Y coordinate ..')
dqc = DataQualityChecker(df=pd.concat([X_train_ycoords, y_train_ycoords], axis=1))
dqc.print_completeness()

logging.info('Data quality check - Test set Y coordinate ..')
dqc = DataQualityChecker(df=pd.concat([X_test_ycoords, y_test_ycoords], axis=1))
dqc.print_completeness()
logging.info('.. Done')

xcoords_model = Modeler(model_type=stg.COORDS_MODEL_TYPE,
                        hyperparameters=stg.X_PROJ_MODEL_BASE_HYPERPARAMS,
                        target=stg.X_PROJ_TARGET, features=stg.X_PROJ_FEATURES)
ycoords_model = Modeler(model_type=stg.COORDS_MODEL_TYPE,
                        hyperparameters=stg.Y_PROJ_MODEL_BASE_HYPERPARAMS,
                        target=stg.Y_PROJ_TARGET, features=stg.Y_PROJ_FEATURES)

if stg.BOOL_TRAIN_COORDS_MODEL:
    if stg.BOOL_COORDS_RS:
        logging.info('Cross-validation ..')
        xcoords_model.perform_random_search_cv(training_data=pd.concat([X_train_xcoords, y_train_xcoords], axis=1),
                                               score='neg_mean_squared_error',
                                               param_distributions=stg.X_PROJ_RANDOM_SEARCH_HYPERPARAMS,
                                               n_jobs=stg.N_JOBS)
        logging.debug('Done for X coordinate')
        ycoords_model.perform_random_search_cv(training_data=pd.concat([X_train_ycoords, y_train_ycoords], axis=1),
                                               score='neg_mean_squared_error',
                                               param_distributions=stg.Y_PROJ_RANDOM_SEARCH_HYPERPARAMS,
                                               n_jobs=stg.N_JOBS)
        logging.debug('Done for Y coordinate')
        logging.info('.. Done')

    logging.info('Model to predict coordinates..')
    xcoords_model.model.set_params(**{'n_jobs': 3})
    xcoords_model.fit(training_data=pd.concat([X_train_xcoords, y_train_xcoords], axis=1))
    logging.debug('X coordinate - Fit ok')
    xcoords_model.save_model(save_modelname=stg.X_PROJ_MODEL_NAME)
    logging.debug('X coordinate - Model saved')

    ycoords_model.model.set_params(**{'n_jobs': 3})
    ycoords_model.fit(training_data=pd.concat([X_train_ycoords, y_train_ycoords], axis=1))
    logging.debug('Y coordinate - Fit ok')
    ycoords_model.save_model(save_modelname=stg.Y_PROJ_MODEL_NAME)
    logging.debug('Y coordinate - Model saved')
    logging.info('.. Done')
else:
    logging.info('Loading latest model ..')
    xcoords_model.load_model(save_modelname=stg.X_PROJ_MODEL_NAME)
    ycoords_model.load_model(save_modelname=stg.Y_PROJ_MODEL_NAME)
    logging.info('.. Done')

"""
x_true = train[stg.X_PROJ_TARGET]
x_pred = train['x_along_team1_axis_lag1']
y_true = train[stg.Y_PROJ_TARGET]
y_pred = train['y_along_team1_axis_lag1']

true = [(x, y) for (x, y) in zip(x_true, y_true)]
pred = [(x, y) for (x, y) in zip(x_pred, y_pred)]

pa = PerformanceAnalyzer(y_true=true, y_pred=pred)
accuracy = pa.compute_accuracy_l2_error()
print('L2 accuracy: {}'.format(accuracy))
"""

logging.info('Performance evaluation ..')
xcoords_pred = xcoords_model.predict(test_data=X_test_xcoords)
ycoords_pred = ycoords_model.predict(test_data=X_test_ycoords)

pred = [(x, y) for (x, y) in zip(xcoords_pred, ycoords_pred)]
true = [(x, y) for (x, y) in zip(y_test_xcoords, y_test_ycoords)]

pa = PerformanceAnalyzer(y_true=true, y_pred=pred)
accuracy = pa.compute_accuracy_l2_error()
logging.info('L2 accuracy: {}'.format(accuracy))
logging.info('.. Done')

logging.info('End of script {}'.format(basename(__file__)))
