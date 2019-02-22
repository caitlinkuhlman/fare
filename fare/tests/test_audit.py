"""Testing for audit module"""


import pytest

#from sklearn import datasets

#from sklearn.utils.testing import assert_equal, assert_almost_equal

from fare.audit import audit_parity
from fare.audit import audit_equality
from fare.audit import audit_calibration
from fare.audit import generate_diagnostics


@pytest.mark.audit_parity
def test_audit_parity():
    assert True

