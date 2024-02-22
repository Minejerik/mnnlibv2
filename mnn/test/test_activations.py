import pytest

def test_relu():
    from mnn.activations import relu
    assert relu(0) == 0
    assert relu(1) == 1
    assert relu(-1) == 0
    assert relu(0.5) == 0.5
    assert relu(-0.5) == 0
  
def test_binary():
    from mnn.activations import binary
    assert binary(0) == 0
    assert binary(1) == 1
    assert binary(-1) == 0
    assert binary(0.5) == 1
    assert binary(-0.5) == 0