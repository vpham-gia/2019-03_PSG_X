from os.path import join
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier
from xgboost.sklearn import XGBClassifier, XGBRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV

import settings as stg


class Modeler():
    """Predictor for players and for next event.

    Relies on a RandomForestRegressor.
    Tries to predict player_id from game stats and next event from previous events

    Attributes
    ----------
    model: sklearn model
    target: string
    features: list

    """

    MODEL_DICT = {'knn': KNeighborsClassifier(),
                  'et_classif': ExtraTreesClassifier(),
                  'rf_classif': RandomForestClassifier(),
                  'rf_reg': RandomForestRegressor(),
                  'xgb_classif': XGBClassifier(),
                  'xgb_reg': XGBRegressor()}

    def __init__(self, model_type, hyperparameters, target, features):
        """Init class.

        model_type: string, ['knn', et_classif, 'rf_classif', 'rf_reg', 'xgb_classif', 'xgb_reg']
        hyperparameters: dict
        target: string
        features: list

        """
        self.model = self.MODEL_DICT[model_type].set_params(**hyperparameters)
        self.target = target
        self.features = features

    def perform_random_search_cv(self, training_data, param_distributions, score, n_jobs):
        """Perform Random search on hyperparameters.

        Parameters
        ----------
        param_distributions: dict
        score: string, ['accuracy', 'l2']

        """
        random_search = RandomizedSearchCV(estimator=self.model,
                                           param_distributions=param_distributions,
                                           n_iter=30, scoring=score, cv=3,
                                           n_jobs=n_jobs)

        random_search.fit(training_data[self.features], training_data[self.target])

        best_estimator = random_search.best_estimator_
        print('Best estimator: {}'.format(best_estimator.get_params()))
        self.model = best_estimator
        return self

    def fit(self, training_data):
        """Override fit method."""
        X_train = training_data[self.features]
        y_train = training_data[self.target]

        predictor = self.model.fit(X_train, y_train)
        self.predictor = predictor

        return self

    def save_model(self, save_modelname):
        """Save model."""
        joblib.dump(self.predictor,
                    join(stg.MODELS_DIR, save_modelname))
        pass

    def load_model(self, save_modelname):
        """Load model."""
        self.predictor = joblib.load(join(stg.MODELS_DIR, save_modelname))
        return self

    def predict(self, test_data):
        """Override predict method."""
        return self.predictor.predict(test_data)
