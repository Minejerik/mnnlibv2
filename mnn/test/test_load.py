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

def test_network_error():
    from mnn.load import load
    import os
    t = os.path.dirname(__file__)
    l = load(f"{t}/test.mnn")
    net = l.load()
    with pytest.raises(Exception):
        net.run([1.0,2.0,3.0,4.0])

def test_network_run():
    MODEL_OUTPUT = [-4.070815362850945, -5.147042327990695, 1.5272360153526403]
    from mnn.load import load
    import os
    t = os.path.dirname(__file__)
    l = load(f"{t}/test.mnn")
    net = l.load()
    assert net.run([1.0,2.0,3.0]) == pytest.approx(MODEL_OUTPUT)