"""
Neural network regressor that cannot extrapolate to impossible prices.

The plain MLPRegressor made extreme predictions for parcels whose features lay outside the
training range (e.g. a very large area or a distance of 111 km), because ReLU networks continue
linearly beyond the data. BoundedMLP fixes this in two ways:
  1. the target (log value) is standardised before training, which makes optimisation stable;
  2. predictions are clipped to the range of log values seen in training, the same bound that
     tree-based models have by construction.
All MLPRegressor hyperparameters are unchanged, so it tunes exactly like the original.
"""
import numpy as np
from sklearn.neural_network import MLPRegressor


class BoundedMLP(MLPRegressor):
    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        self.y_mean_, self.y_std_ = y.mean(), y.std() or 1.0
        self.y_min_, self.y_max_ = y.min(), y.max()
        return super().fit(X, (y - self.y_mean_) / self.y_std_)

    def predict(self, X):
        p = super().predict(X) * self.y_std_ + self.y_mean_
        return np.clip(p, self.y_min_, self.y_max_)
