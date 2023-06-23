import sys 
sys.path.append('./python')

import TransFTrain as train
import TransFTrain.nn as nn

import unittest
import numpy as np

class TestSequential(unittest.TestCase): 
    def get_tensor(self, *shape, entropy=1):
        np.random.seed(np.prod(shape) * len(shape) * entropy)
        return train.Tensor(np.random.randint(0, 100, size=shape) / 20, dtype="float32")

    def get_int_tensor(self, *shape, low=0, high=10, entropy=1):
        np.random.seed(np.prod(shape) * len(shape) * entropy)
        return train.Tensor(np.random.randint(low, high, size=shape))

    def sequential_forward(self, batches=3):
        np.random.seed(42)
        f = nn.Sequential(nn.Linear(5, 8), nn.ReLU(), nn.Linear(8, 5))
        x = self.get_tensor(batches, 5)
        return f(x).cached_data

    def sequential_backward(self, batches=3):
        np.random.seed(42)
        f = nn.Sequential(nn.Linear(5, 8), nn.ReLU(), nn.Linear(8, 5))
        x = self.get_tensor(batches, 5)
        f(x).sum().backward()
        return x.grad.cached_data
    
    def test_nn_sequential_forward_1(self):
        np.testing.assert_allclose(self.sequential_forward(batches=3),
            np.array([[ 3.296263,  0.057031,  2.97568 , -4.618432, -0.902491],
                [ 2.465332, -0.228394,  2.069803, -3.772378, -0.238334],
                [ 3.04427 , -0.25623 ,  3.848721, -6.586399, -0.576819]], dtype=np.float32), rtol=1e-5, atol=1e-5)

    def test_nn_sequential_backward_1(self):
        np.testing.assert_allclose(self.sequential_backward(batches=3),
            np.array([[ 0.802697, -1.0971  ,  0.120842,  0.033051,  0.241105],
                [-0.364489,  0.651385,  0.482428,  0.925252, -1.233545],
                [ 0.802697, -1.0971  ,  0.120842,  0.033051,  0.241105]], dtype=np.float32), rtol=1e-5, atol=1e-5)

if "__main__" == __name__:
    unittest.main()