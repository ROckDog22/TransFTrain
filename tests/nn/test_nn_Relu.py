import sys 
sys.path.append('./python')

import TransFTrain as train
import TransFTrain.nn as nn

import unittest
import numpy as np

class TestRelu(unittest.TestCase): 
    def get_tensor(self, *shape, entropy=1):
        np.random.seed(np.prod(shape) * len(shape) * entropy)
        return train.Tensor(np.random.randint(0, 100, size=shape) / 20, dtype="float32")

    def get_int_tensor(self, *shape, low=0, high=10, entropy=1):
        np.random.seed(np.prod(shape) * len(shape) * entropy)
        return train.Tensor(np.random.randint(low, high, size=shape))

    def relu_forward(self, *shape):
        f = train.nn.ReLU()
        x = self.get_tensor(*shape)
        return f(x).cached_data

    def relu_backward(self, *shape):
        f = train.nn.ReLU()
        x = self.get_tensor(*shape)
        (f(x)**2).sum().backward()
        return x.grad.cached_data
    
    def test_nn_relu_forward_1(self):
        np.testing.assert_allclose(self.relu_forward(2, 2),
            np.array([[3.35, 4.2 ],
            [0.25, 4.5 ]], dtype=np.float32), rtol=1e-5, atol=1e-5)

    def test_nn_relu_backward_1(self):
        np.testing.assert_allclose(self.relu_backward(3, 2),
            np.array([[7.5, 2.7],
            [0.6, 0.2],
            [0.3, 6.7]], dtype=np.float32), rtol=1e-5, atol=1e-5)

if "__main__" == __name__:
    unittest.main()