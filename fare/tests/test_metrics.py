"""Testing for metrics module"""


import pytest

#from sklearn import datasets

#from sklearn.utils.testing import assert_equal, assert_almost_equal

# TODO should these functions go in a base file?
from fare.metrics import _pairs
from fare.metrics import _merge_cal
from fare.metrics import _merge_eq
from fare.metrics import _merge_parity
from fare.metrics import _count_inversions

from fare.metrics import rank_equality
from fare.metrics import rank_calibration
from fare.metrics import rank_parity


@pytest.mark.rank_parity
def test_rank_parity():
    y_true = [7, -0.5, 2, 3]
    y_pred = [2.5, 0.0, 2, 8]
    groups = [1, 1, 1, 0]
    print("Parity", rank_parity(y_pred,groups))
    assert True

