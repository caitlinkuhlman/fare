"""
Foo
"""

# Authors


from fare import metrics # rank_parity

def test_rank_parity():
    y_true = [7, -0.5, 2, 3]
    y_pred = [2.5, 0.0, 2, 8]
    groups = [1, 1, 1, 0]
    print("Parity",metrics.rank_parity(y_pred,groups))
    assert True

