import unittest
from mnn.utils import mult_lists, mae_loss

class TestUtils(unittest.TestCase):
    def test_mult_lists(self):
        list1 = [1, 2, 3]
        list2 = [4, 5, 6]
        self.assertEqual(mult_lists(list1, list2), [4, 10, 18])
        
    def test_mae_loss(self):
        y_true = [1, 2, 3]
        y_pred = [4, 5, 6]
        self.assertEqual(mae_loss(y_true, y_pred), 3)
        
class TestActivations(unittest.TestCase):
    def test_relu(self):
        from mnn.activations import relu
        self.assertEqual(relu(-100), 0)
        self.assertEqual(relu(100), 100)
        


if __name__ == '__main__':
    unittest.main()