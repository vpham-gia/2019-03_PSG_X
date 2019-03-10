from os.path import join
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

import settings as stg


class PlayerPredictor():
    """Predictor for players.

    Relies on a RandomForestRegressor.
    Tries to predict player_id from game stats

    Attributes
    ----------
    model: sklearn model
    target: string
    features: list

    """

    MODEL_DICT = {'rf': RandomForestClassifier(),
                  'knn': KNeighborsClassifier()}

    def __init__(self, model_type, hyperparameters, target, features):
        """Init class.

        model_type: string, ['rf', 'knn']
        hyperparameters: dict
        target: string
        features: list

        """
        self.model = self.MODEL_DICT[model_type].set_params(**hyperparameters)
        self.target = target
        self.features = features

    def fit(self, training_data):
        """Override fit method."""
        X_train = training_data[self.FEATURES]
        y_train = training_data[self.TARGET]

        predictor = self.model.fit(X_train, y_train)
        self.predictor = predictor

        return self

    def save_model(self):
        """Save model."""
        joblib.dump(self.predictor,
                    join(stg.MODELS_DIR, stg.MODEL_NAME))
        pass

    def load_mode(self):
        """Load model."""
        self.predictor = joblib.load(join(stg.MODELS_DIR, stg.MODEL_NAME))
        return self

    def predict(self, test_data):
        """Override predict method."""
        return self.predictor.predict(test_data)
