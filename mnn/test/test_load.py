import pytest

def test_layer_loading():
    from mnn.load import load
    import os
    t = os.path.dirname(__file__)
    l = load(f"{t}/test.mnn")
    net = l.load()
    assert len(net.layers) == 3

def test_layer_length():
    from mnn.load import load
    import os
    t = os.path.dirname(__file__)
    l = load(f"{t}/test.mnn")
    net = l.load()
    assert len(net.layers[0].neurons) == 5

def test_layer_activations():
    from mnn.load import load
    import os
    t = os.path.dirname(__file__)
    l = load(f"{t}/test.mnn")
    net = l.load()
    assert net.layers[0].neurons[0].activation(5) == 5