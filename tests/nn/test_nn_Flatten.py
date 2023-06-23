import sys 
sys.path.append('./python')

import TransFTrain as train
import TransFTrain.nn as nn

import unittest
import numpy as np

class TestFlatten(unittest.TestCase): 
    def get_tensor(self, *shape, entropy=1):
        np.random.seed(np.prod(shape) * len(shape) * entropy)
        return train.Tensor(np.random.randint(0, 100, size=shape) / 20, dtype="float32")

    def get_int_tensor(self, *shape, low=0, high=10, entropy=1):
        np.random.seed(np.prod(shape) * len(shape) * entropy)
        return train.Tensor(np.random.randint(low, high, size=shape))
    
    def flatten_forward(self,*shape):
        x = self.get_tensor(*shape)
        tform = train.nn.Flatten()
        return tform(x).cached_data

    def flatten_backward(self, *shape):
        x = self.get_tensor(*shape)
        tform = train.nn.Flatten()
        (tform(x)**2).sum().backward()
        return x.grad.cached_data
    
    def test_nn_flatten_forward_1(self):
        np.testing.assert_allclose(self.flatten_forward(3,3), np.array([[2.1 , 0.95, 3.45],
        [3.1 , 2.45, 2.3 ],
        [3.3 , 0.4 , 1.2 ]], dtype=np.float32), rtol=1e-5, atol=1e-5)


    def test_nn_flatten_forward_2(self):
        np.testing.assert_allclose(self.flatten_forward(3,3,3), np.array([[3.35, 3.25, 2.8 , 2.3 , 3.75, 3.75, 3.35, 2.45, 2.1 ],
        [1.65, 0.15, 4.15, 2.8 , 2.1 , 0.5 , 2.6 , 2.25, 3.25],
        [2.4 , 4.55, 4.75, 0.75, 3.85, 0.05, 4.7 , 1.7 , 4.7 ]], dtype=np.float32), rtol=1e-5, atol=1e-5)


    def test_nn_flatten_forward_3(self):
        np.testing.assert_allclose(self.flatten_forward(1,2,3,4), np.array([[4.2 , 4.5 , 1.9 , 4.85, 4.85, 3.3 , 2.7 , 3.05, 0.3 , 3.65, 3.1 ,
            0.1 , 4.5 , 4.05, 3.05, 0.15, 3.  , 1.65, 4.85, 1.3 , 3.95, 2.9 ,
            1.2 , 1.  ]], dtype=np.float32), rtol=1e-5, atol=1e-5)


    def test_nn_flatten_forward_4(self):
        np.testing.assert_allclose(self.flatten_forward(3,3,4,4), np.array([[0.95, 1.1 , 1.  , 1.  , 4.9 , 0.25, 1.6 , 0.35, 1.5 , 3.4 , 1.75,
            3.4 , 4.8 , 1.4 , 2.35, 3.2 , 1.65, 1.9 , 3.05, 0.35, 3.15, 4.05,
            3.3 , 2.2 , 2.5 , 1.5 , 3.25, 0.65, 3.05, 0.75, 3.25, 2.55, 0.55,
            0.25, 3.65, 3.4 , 0.05, 1.4 , 0.75, 1.55, 4.45, 0.2 , 3.35, 2.45,
            3.45, 4.75, 2.45, 4.3 ],
        [1.  , 0.2 , 0.4 , 0.7 , 4.9 , 4.2 , 2.55, 3.15, 1.2 , 3.8 , 1.35,
            1.85, 3.15, 2.7 , 1.5 , 1.35, 4.85, 4.2 , 1.5 , 1.75, 0.8 , 4.3 ,
            4.2 , 4.85, 0.  , 3.75, 0.9 , 0.  , 3.35, 1.05, 2.2 , 0.75, 3.6 ,
            2.  , 1.2 , 1.9 , 3.45, 1.6 , 3.95, 4.45, 4.55, 4.75, 3.7 , 0.3 ,
            2.45, 3.75, 0.9 , 2.2 ],
        [4.95, 1.05, 2.4 , 4.05, 3.75, 1.95, 0.65, 4.9 , 4.3 , 2.5 , 1.9 ,
            1.75, 2.05, 3.95, 0.8 , 0.  , 0.8 , 3.45, 1.55, 0.3 , 1.5 , 2.9 ,
            2.15, 2.15, 3.3 , 3.2 , 4.3 , 3.7 , 0.4 , 1.7 , 0.35, 1.9 , 1.8 ,
            4.3 , 4.7 , 4.05, 3.65, 1.1 , 1.  , 2.7 , 3.95, 2.3 , 2.6 , 3.5 ,
            0.75, 4.3 , 3.  , 3.85]], dtype=np.float32), rtol=1e-5, atol=1e-5)


    def test_nn_flatten_backward_1(self):
        np.testing.assert_allclose(self.flatten_backward(3,3), np.array([[4.2, 1.9, 6.9],
        [6.2, 4.9, 4.6],
        [6.6, 0.8, 2.4]], dtype=np.float32), rtol=1e-5, atol=1e-5)

    def test_nn_flatten_backward_2(self):
        np.testing.assert_allclose(self.flatten_backward(3,3,3), np.array([[[6.7, 6.5, 5.6],
            [4.6, 7.5, 7.5],
            [6.7, 4.9, 4.2]],

        [[3.3, 0.3, 8.3],
            [5.6, 4.2, 1. ],
            [5.2, 4.5, 6.5]],

        [[4.8, 9.1, 9.5],
            [1.5, 7.7, 0.1],
            [9.4, 3.4, 9.4]]], dtype=np.float32), rtol=1e-5, atol=1e-5)


    def test_nn_flatten_backward_3(self):
        np.testing.assert_allclose(self.flatten_backward(2,2,2,2), np.array([[[[6.8, 3.8],
            [5.4, 5.1]],

            [[8.5, 4.8],
            [3.1, 1. ]]],


        [[[9.3, 0.8],
            [3.4, 1.6]],

            [[9.4, 3.6],
            [6.6, 7. ]]]], dtype=np.float32), rtol=1e-5, atol=1e-5)

    def test_nn_flatten_backward_4(self):
        np.testing.assert_allclose(self.flatten_backward(1,2,3,4), np.array([[[[8.4, 9. , 3.8, 9.7],
            [9.7, 6.6, 5.4, 6.1],
            [0.6, 7.3, 6.2, 0.2]],

            [[9. , 8.1, 6.1, 0.3],
            [6. , 3.3, 9.7, 2.6],
            [7.9, 5.8, 2.4, 2. ]]]], dtype=np.float32), rtol=1e-5, atol=1e-5)


    def test_nn_flatten_backward_5(self):
        np.testing.assert_allclose(self.flatten_backward(2,2,4,3), np.array([[[[9.8, 7.1, 5.4],
            [4. , 6.2, 5.7],
            [7.2, 2. , 2.4],
            [8.9, 4.9, 3.3]],

            [[9. , 9.8, 5.9],
            [7.1, 2.7, 9.6],
            [8.5, 9.3, 5.8],
            [3.1, 9. , 6.7]]],


        [[[7.4, 8.6, 6.9],
            [8.2, 5.3, 8.7],
            [8.8, 8.7, 4. ],
            [3.9, 1.8, 2.7]],

            [[5.7, 6.2, 0. ],
            [6. , 0. , 0.3],
            [2. , 0.1, 2.7],
            [2.1, 0.1, 6.7]]]], dtype=np.float32), rtol=1e-5, atol=1e-5)
        

if "__main__" == __name__:
    unittest.main()
