from sklearn.metrics import accuracy_score
from numpy.linalg import norm
import numpy as np


class PerformanceAnalyzer():
    """Class to compute performance analysis.

    Attributes
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels.
    y_pred : 1d array-like, or label indicator array / sparse matrix
        Predicted labels, as returned by a classifier.

    """

    def __init__(self, y_true, y_pred):
        """Init."""
        self.y_true = y_true
        self.y_pred = y_pred

    def compute_classification_accuracy(self):
        """Classification accuracy score."""
        return accuracy_score(y_true=self.y_true, y_pred=self.y_pred)

    def compute_accuracy_l2_error(self):
        """Metric for position prediction."""
        MAX_ERROR = np.sqrt(100 * 100 + 100 * 100)
        return (1 - self.compute_avg_l2_error() / MAX_ERROR)

    def compute_avg_l2_error(self):
        """Metric for position prediction."""
        l2_distance = norm(np.subtract(self.y_true, self.y_pred), axis=1)
        return np.mean(l2_distance)
