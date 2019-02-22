import pytest

from sklearn.utils.estimator_checks import check_estimator

from fare import TemplateEstimator
from fare import TemplateClassifier
from fare import TemplateTransformer


@pytest.mark.parametrize(
    "Estimator", [TemplateEstimator, TemplateTransformer, TemplateClassifier]
)
def test_all_estimators(Estimator):
    return check_estimator(Estimator)
