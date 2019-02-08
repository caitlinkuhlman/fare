import pytest

from sklearn.utils.estimator_checks import check_estimator

from FARE import TemplateEstimator
from FARE import TemplateClassifier
from FARE import TemplateTransformer


@pytest.mark.parametrize(
    "Estimator", [TemplateEstimator, TemplateTransformer, TemplateClassifier]
)
def test_all_estimators(Estimator):
    return check_estimator(Estimator)
