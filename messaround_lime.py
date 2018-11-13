import collections
import copy
from functools import partial
import json
import warnings

import numpy as np
import sklearn
import sklearn.preprocessing
from sklearn.utils import check_random_state
from sklearn.linear_model import Ridge


def data_inverse(scaler, data_row, num_samples, random_state):
    """Generates a neighborhood around a prediction.
    For numerical features, perturb them by sampling from a Normal(0,1) and
    doing the inverse operation of mean-centering and scaling, according to
    the means and stds in the training data. For categorical features,
    perturb by sampling according to the training distribution, and making
    a binary feature that is 1 when the value is the same as the instance
    being explained.
    Args:
        data_row: 1d numpy array, corresponding to a row
        num_samples: size of the neighborhood to learn the linear model
    Returns:
        A tuple (data, inverse), where:
            data: dense num_samples * K matrix, where categorical features
            are encoded with either 0 (not equal to the corresponding value
            in data_row) or 1. The first row is the original instance.
            inverse: same as data, except the categorical features are not
            binary, but categorical (as the original data)
    """
    # no discretizer
    data = check_random_state(random_state).normal(
        0, 1, num_samples * data_row.shape[0]).reshape(
        num_samples, data_row.shape[0])
    data = data * scaler.scale_ + data_row
    data[0] = data_row.copy()
    return data



def kernel(d, kernel_width):
    return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

kernel_width = np.sqrt(train_x.shape[1]) * .75
kernel_width = float(kernel_width)
kernel_fn = partial(kernel, kernel_width=kernel_width)

scaled_data = (data - scaler.mean_) / scaler.scale_
distances = sklearn.metrics.pairwise_distances(
    scaled_data,
    scaled_data[0].reshape(1, -1),
    metric=distance_metric
).ravel()


def feature_selection(data, labels, weights, num_features):
    clf = Ridge(alpha=0, fit_intercept=True,
                random_state=7)
    clf.fit(data, labels, sample_weight=weights)
    feature_weights = sorted(zip(range(data.shape[0]),
                                 clf.coef_ * data[0]),
                             key=lambda x: np.abs(x[1]),
                             reverse=True)
    return np.array([x[0] for x in feature_weights[:num_features]])


def explain_instance_with_data(kernel_fn,
                               neighborhood_data,
                               neighborhood_labels,
                               distances,
                               num_features):
    """Takes perturbed data, labels and distances, returns explanation.
    Args:
        neighborhood_data: perturbed data, 2d array. first element is
                           assumed to be the original data point.
        neighborhood_labels: corresponding perturbed labels. should have as
                             many columns as the number of possible labels.
        distances: distances to original data point.
        label: label for which we want an explanation
        num_features: maximum number of features in explanation
    Returns:
        (intercept, exp, score):
        intercept is a float.
        exp is a sorted list of tuples, where each tuple (x,y) corresponds
        to the feature id (x) and the local weight (y). The list is sorted
        by decreasing absolute value of y.
        score is the R^2 value of the returned explanation
    """

    weights = kernel_fn(distances)
    labels_column = neighborhood_labels
    used_features = feature_selection(neighborhood_data,
                                           labels_column,
                                           weights,
                                           num_features)

    model_regressor = Ridge(alpha=1, fit_intercept=True, random_state=7)
    easy_model = model_regressor
    easy_model.fit(neighborhood_data[:, used_features],
                   labels_column, sample_weight=weights)
    prediction_score = easy_model.score(
        neighborhood_data[:, used_features],
        labels_column, sample_weight=weights)

    local_pred = easy_model.predict(neighborhood_data[0, used_features].reshape(1, -1))
    return (easy_model,
            sorted(zip(used_features, easy_model.coef_),
                   key=lambda x: np.abs(x[1]), reverse=True),
            prediction_score, local_pred)