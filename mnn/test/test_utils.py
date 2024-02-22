import pytest

def test_mult_lists():
    from mnn.utils import mult_lists
    inp1 = [1, 2, 3]
    inp2 = [4, 5, 6]
    assert mult_lists(inp1, inp2) == [4, 10, 18]
    
def test_empty_mult_lists():
    from mnn.utils import mult_lists
    inp1 = []
    inp2 = []
    assert mult_lists(inp1, inp2) == []
    
def test_mae_loss():
    from mnn.utils import mae_loss
    inp1 = [1, 2, 3]
    inp2 = [4, 5, 6]
    assert mae_loss(inp1, inp2) == 3
    
def test_list_avg():
    from mnn.utils import list_avg
    inp = [1, 2, 3]
    assert list_avg(inp) == 2