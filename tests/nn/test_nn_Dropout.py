import sys 
sys.path.append('./python')

import TransFTrain as train
import TransFTrain.nn as nn

import unittest
import numpy as np

class TestDropout(unittest.TestCase): 
    def get_tensor(self, *shape, entropy=1):
        np.random.seed(np.prod(shape) * len(shape) * entropy)
        return train.Tensor(np.random.randint(0, 100, size=shape) / 20, dtype="float32")

    def get_int_tensor(self, *shape, low=0, high=10, entropy=1):
        np.random.seed(np.prod(shape) * len(shape) * entropy)
        return train.Tensor(np.random.randint(low, high, size=shape))

    def dropout_forward(self, shape, prob=0.5):
        np.random.seed(3)
        x = self.get_tensor(*shape)
        f = nn.Dropout(prob)
        return f(x).cached_data

    def dropout_backward(self, shape, prob=0.5):
        np.random.seed(3)
        x = self.get_tensor(*shape)
        f = nn.Dropout(prob)
        y = f(x).sum()
        y.backward()
        return x.grad.cached_data

    def test_nn_dropout_forward_1(self):
        np.testing.assert_allclose(self.dropout_forward((2, 3), prob=0.45),
            np.array([[6.818182 , 0. , 0. ],
            [0.18181819, 0. , 6.090909 ]], dtype=np.float32), rtol=1e-5, atol=1e-5)

    def test_nn_dropout_backward_1(self):
        np.testing.assert_allclose(self.dropout_backward((2, 3), prob=0.26),
            np.array([[1.3513514, 0. , 0. ],
            [1.3513514, 0. , 1.3513514]], dtype=np.float32), rtol=1e-5, atol=1e-5)

if "__main__" == __name__:
    unittest.main()