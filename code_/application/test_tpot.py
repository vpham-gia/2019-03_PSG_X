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

from tpot import TPOTClassifier
pipeline_optimizer = TPOTClassifier(generations=100, population_size=100, cv=3,
                                    random_state=42, verbosity=2, max_time_mins=1)
pipeline_optimizer.fit(X_train, y_train)
print('Score: {}'.format(pipeline_optimizer.score(X_test, y_test)))
pipeline_optimizer.export('tpot_exported_pipeline.py')

logging.info('End of script {}'.format(basename(__file__)))
