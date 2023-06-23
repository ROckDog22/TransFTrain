import sys 
sys.path.append('./python')

import TransFTrain as train
import TransFTrain.nn as nn

import unittest
import numpy as np

class TestResidual(unittest.TestCase): 
    def get_tensor(self, *shape, entropy=1):
        np.random.seed(np.prod(shape) * len(shape) * entropy)
        return train.Tensor(np.random.randint(0, 100, size=shape) / 20, dtype="float32")

    def get_int_tensor(self, *shape, low=0, high=10, entropy=1):
        np.random.seed(np.prod(shape) * len(shape) * entropy)
        return train.Tensor(np.random.randint(low, high, size=shape))

    def residual_forward(self, shape=(5,5)):
        np.random.seed(42)
        f = nn.Residual(nn.Sequential(nn.Linear(*shape), nn.ReLU(), nn.Linear(*shape[::-1])))
        x = self.get_tensor(*shape[::-1])
        return f(x).cached_data

    def residual_backward(self, shape=(5,5)):
        np.random.seed(42)
        f = nn.Residual(nn.Sequential(nn.Linear(*shape), nn.ReLU(), nn.Linear(*shape[::-1])))
        x = self.get_tensor(*shape[::-1])
        f(x).sum().backward()
        return x.grad.cached_data

    def test_nn_residual_forward_1(self):
        np.testing.assert_allclose(self.residual_forward(),
            np.array([[ 0.4660964 ,  3.8619597,  -3.637068  ,  3.7489638,   2.4931884 ],
                    [-3.3769124 ,  2.5409935,  -2.7110925 ,  4.9782896,  -3.005401  ],
                    [-3.0222898 ,  3.796795 ,  -2.101042  ,  6.785948 ,   0.9347453 ],
                    [-2.2496533 ,  3.635599 ,  -2.1818666 ,  5.6361046,   0.9748006 ],
                    [-0.03458184,  0.0823682,  -0.06686163,  1.9169499,   1.2638961 ]],
            dtype=np.float32), rtol=1e-5, atol=1e-5)

    def test_nn_residual_backward_1(self):
        np.testing.assert_allclose(self.residual_backward(),
            np.array([[ 0.24244219, -0.19571924, -0.08556509,  0.9191598,   1.6787351 ],
                        [ 0.24244219, -0.19571924, -0.08556509,  0.9191598,   1.6787351 ],
                        [ 0.24244219, -0.19571924, -0.08556509,  0.9191598,   1.6787351 ],
                        [ 0.24244219, -0.19571924, -0.08556509,  0.9191598,   1.6787351 ],
                        [ 0.24244219, -0.19571924, -0.08556509,  0.9191598,   1.6787351 ]],
            dtype=np.float32), rtol=1e-5, atol=1e-5)
    
if "__main__" == __name__:
    unittest.main()