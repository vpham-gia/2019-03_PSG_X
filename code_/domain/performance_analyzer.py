from sklearn.metrics import accuracy_score


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

    def compute_l2_error(self):
        """Metric for position prediction."""
        pass
