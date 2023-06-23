import sys 
sys.path.append('./python')

import TransFTrain as train
import TransFTrain.nn as nn

import unittest
import numpy as np

class TestLayerNorm1d(unittest.TestCase): 
    def get_tensor(self, *shape, entropy=1):
        np.random.seed(np.prod(shape) * len(shape) * entropy)
        return train.Tensor(np.random.randint(0, 100, size=shape) / 20, dtype="float32")

    def get_int_tensor(self, *shape, low=0, high=10, entropy=1):
        np.random.seed(np.prod(shape) * len(shape) * entropy)
        return train.Tensor(np.random.randint(low, high, size=shape))

    def layernorm_forward(self, shape, dim):
        f = train.nn.LayerNorm1d(dim)
        x = self.get_tensor(*shape)
        return f(x).cached_data

    def layernorm_backward(self, shape, dims):
        f = train.nn.LayerNorm1d(dims)
        x = self.get_tensor(*shape)
        (f(x)**4).sum().backward()
        return x.grad.cached_data
    
    def test_nn_layernorm_forward_1(self):
        np.testing.assert_allclose(self.layernorm_forward((3, 3), 3),
            np.array([[-0.06525002, -1.1908097 ,  1.2560595 ],
        [ 1.3919864 , -0.47999576, -0.911992  ],
        [ 1.3628436 , -1.0085043 , -0.3543393 ]], dtype=np.float32), rtol=1e-5, atol=1e-5)

    def test_nn_layernorm_forward_2(self):
        np.testing.assert_allclose(self.layernorm_forward((2,10), 10),
            np.array([[ 0.8297899 ,  1.6147263 , -1.525019  , -0.4036814 ,  0.306499  ,
            0.08223152,  0.6429003 , -1.3381294 ,  0.8671678 , -1.0764838 ],
        [-1.8211555 ,  0.39098236, -0.5864739 ,  0.853988  , -0.3806936 ,
            1.2655486 ,  0.33953735,  1.522774  , -0.8951442 , -0.68936396]], dtype=np.float32), rtol=1e-5, atol=1e-5)

    def test_nn_layernorm_forward_3(self):
        np.testing.assert_allclose(self.layernorm_forward((1,5), 5),
            np.array([[-1.0435007 , -0.8478443 ,  0.7500162 , -0.42392215,  1.565251  ]], dtype=np.float32), rtol=1e-5, atol=1e-5)


    def test_nn_layernorm_backward_1(self):
        np.testing.assert_allclose(self.layernorm_backward((3, 3), 3),
            np.array([[-2.8312206e-06, -6.6757202e-05,  6.9618225e-05],
        [ 1.9950867e-03, -6.8092346e-04, -1.3141632e-03],
        [ 4.4703484e-05, -3.2544136e-05, -1.1801720e-05]], dtype=np.float32), rtol=1e-5, atol=1e-5)

    def test_nn_layernorm_backward_2(self):
        np.testing.assert_allclose(self.layernorm_backward((2,10), 10),
            np.array([[-2.301574  ,  4.353944  , -1.9396116 ,  2.4330146 , -1.1070801 ,
            0.01571643, -2.209449  ,  0.49513134, -2.261348  ,  2.5212562 ],
        [-9.042961  , -2.6184766 ,  4.5592957 , -4.2109876 ,  3.4247458 ,
            -1.9075732 , -2.2689414 ,  2.110825  ,  5.044025  ,  4.910048  ]], dtype=np.float32), rtol=1e-5, atol=1e-5)

    def test_nn_layernorm_backward_3(self):
        np.testing.assert_allclose(self.layernorm_backward((1,5), 5),
            np.array([[ 0.150192,  0.702322, -3.321343,  0.31219 ,  2.156639]], dtype=np.float32), rtol=1e-5, atol=1e-5)


    def test_nn_layernorm_backward_4(self):
        np.testing.assert_allclose(self.layernorm_backward((5,1), 1),
            np.array([[ 0 ],  [0],[0],[0],[0]], dtype=np.float32), rtol=1e-5, atol=1e-5)
        

if "__main__" == __name__:
    unittest.main()