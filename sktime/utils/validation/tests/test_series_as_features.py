#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = ["Markus Löning"]
__all__ = [
    "test_check_X_bad_input_args"
]

import numpy as np
import pandas as pd
import pytest
from sktime.utils._testing import make_classification_problem
from sktime.utils.validation.series_as_features import check_X
from sktime.utils.validation.series_as_features import check_X_y

bad_input_args = [
    [0, 1, 2],  # list
    np.empty((3, 2)),  # 2d np.array
    np.empty(2),  # 1d np.array
    np.empty((3, 2, 3, 2)),  # 4d np.array
    pd.DataFrame(np.empty((2, 3)))  # non-nested pd.DataFrame
]
y = pd.Series()


@pytest.mark.parametrize("X", bad_input_args)
def test_check_X_bad_input_args(X):
    with pytest.raises(ValueError):
        check_X(X)
        check_X_y(X, y)


def test_check_X_enforce_min_instances():
    X, _ = make_classification_problem(n_instances=2)
    with pytest.raises(ValueError):
        check_X(X, enforce_min_instances=3)


def test_check_X_enforce_univariate():
    X, _ = make_classification_problem(n_columns=2)
    with pytest.raises(ValueError):
        check_X(X, enforce_univariate=True)


def test_check_X_enforce_min_columns():
    X, _ = make_classification_problem(n_columns=2)
    with pytest.raises(ValueError):
        check_X(X, enforce_min_columns=3)
