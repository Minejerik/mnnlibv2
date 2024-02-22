import pytest

def test_load_csv():
    from mnn.loader import load_csv
    with pytest.raises(DeprecationWarning):
        load_csv("test.csv")