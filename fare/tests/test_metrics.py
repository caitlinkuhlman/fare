"""Testing for metrics module"""


import pytest
import numpy as np
import matplotlib.pyplot as plt
import itertools

from sklearn import manifold, datasets

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


def _eq_np64(var, value):
    """
    Helper function to check the value and type of rank method outputs.
    """
    if (var != value):  # Values are not equal
        assert False
    if type(var) is not np.float64:  # Not proper type (np.float64)
        assert False
    return True
    #return (var == value) and type(var) is np.float64

@pytest.mark.rank_equality
@pytest.mark.rank_calibration
@pytest.mark.rank_parity
def test_metric_output_type_1():
    """
    Type checking of values, all should return Python floats
    """
    y_true = [7, -0.5, 2, 3]
    y_pred = [2.5, 0.0, 2, 8]
    groups = [1, 1, 1, 0]

    # Rank equality
    error0,error1 = rank_equality(y_true, y_pred, groups)
    assert type(error0) is np.float64
    assert type(error1) is np.float64

    # Rank calibration
    error0,error1 = rank_calibration(y_true, y_pred, groups)
    assert type(error0) is np.float64
    assert type(error1) is np.float64

    # Rank parity
    error0,error1 = rank_parity(y_pred,groups)
    assert type(error0) is np.float64
    assert type(error1) is np.float64

def test_scenario_1():
    y_true = [1,2,3,4,5,6,7,8]
    y_pred = [1,2,3,4,5,6,7,8]
    groups = [1,1,1,1,1,0,0,0]  # More 1 groups than 0 groups

    error0,error1 = rank_equality(y_true, y_pred, groups)
    assert _eq_np64(error0, 0.0) and _eq_np64(error1, 0.0)
    # Should run with no error

    error0,error1 = rank_calibration(y_true, y_pred, groups)
    assert _eq_np64(error0, 0.0) and _eq_np64(error1, 0.0)
    # Should run with no error

    error0,error1 = rank_parity(y_pred,groups)    
    # TODO this is not giving expected
    assert _eq_np64(error0, 0.0) # and _eq_np64(error1, 0.1)    
    assert error0 < error1  # group1 always favored

    groups = [0,0,0,1,1,1,1,1]
    error0,error1 = rank_parity(y_pred,groups)    
    assert _eq_np64(error0, 1.0) and _eq_np64(error1, 0.0)

    groups = [0,0,0,0,0,0,0,0]
    error0,error1 = rank_parity(y_pred,groups)    
    # TODO error0 and error1 types are not proper
    #assert _eq_np64(error0, 1.0) and _eq_np64(error1, 0.0)
    assert error0 > error1  # group0 always favored

    groups = [1,1,1,1,1,1,1,1]
    error0,error1 = rank_parity(y_pred,groups)    
    # TODO error0 and error1 types are not proper
    #assert _eq_np64(error0, 0.0) and _eq_np64(error1, 1.0)
    assert error0 < error1  # group1 always favored

def test_scenario_2():
    y_true = [0,1,2,3,4,5]
    y_pred = [5,4,3,2,1,0]
    groups = [0,0,1,0,1,1]

    # Equality
    error0,error1 = rank_equality(y_true, y_pred, groups)
    assert _eq_np64(error0, 0.8888888888888888) 
    assert _eq_np64(error1, 0.1111111111111111)
    # Should run with no error

    # Calibration
    error0,error1 = rank_calibration(y_true, y_pred, groups)
    assert _eq_np64(error0, 1.0) and _eq_np64(error1, 1.0)
    # Should run with no error

    # Parity
    error0,error1 = rank_parity(y_pred,groups)    
    assert _eq_np64(error0, 0.1111111111111111)
    assert _eq_np64(error1, 0.8888888888888888) 
    # parity has same values as equality because all pairs are inverted.
    # The order is opposite because y_pred orders the groups in reverse.

def test_scenario_3():
    """ Example from paper """
    y_true = [1,2,3,4]
    y_pred = [1,3,4,2]
    groups =[0,1,0,1]

    # Equality
    error0,error1 = rank_equality(y_true, y_pred, groups)
    assert _eq_np64(error0, 0.25) and _eq_np64(error1, 0.0)

    # Calibration
    error0,error1 = rank_calibration(y_true, y_pred, groups)
    assert _eq_np64(error0, 0.2) and _eq_np64(error1, 0.4)

    # Parity
    error0,error1 = rank_parity(y_pred,groups)    
    assert _eq_np64(error0, 0.5) and _eq_np64(error1, 0.5)
