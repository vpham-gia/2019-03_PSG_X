import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFwe, f_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from tpot.builtins import StackingEstimator, ZeroCount
from xgboost import XGBRegressor
from sklearn.preprocessing import FunctionTransformer
from copy import copy

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=42)

# Average CV score on the training set was:-148.31589276097782
exported_pipeline = make_pipeline(
    make_union(
        make_pipeline(
            MaxAbsScaler(),
            RobustScaler(),
            ZeroCount(),
            SelectFwe(score_func=f_regression, alpha=0.038)
        ),
        FunctionTransformer(copy)
    ),
    XGBRegressor(learning_rate=0.1, max_depth=9, min_child_weight=15, n_estimators=100, nthread=1, subsample=1.0)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
