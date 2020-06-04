#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = ["Markus Löning"]
__all__ = []

import pytest
from sktime.utils._testing import make_classification_problem


@pytest.mark.parametrize("n_instances", [1, 3])
@pytest.mark.parametrize("n_columns", [1, 3])
@pytest.mark.parametrize("n_timepoints", [1, 3])
def test_make_classification_problem(n_instances, n_columns, n_timepoints):
    X, y = make_classification_problem(
        n_instances=n_instances,
        n_columns=n_columns,
        n_timepoints=n_timepoints)

    # check dimensions of generated data
    assert len(y) == n_instances
    assert X.shape[:2] == (n_instances, n_columns)
    assert X.iloc[0, 0].shape == (n_timepoints,)
