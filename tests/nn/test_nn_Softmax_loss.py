import sys 
sys.path.append('./python')

import TransFTrain as train
import TransFTrain.nn as nn

import unittest
import numpy as np

class TestSoftmaxLoss(unittest.TestCase): 
    def get_tensor(self, *shape, entropy=1):
        np.random.seed(np.prod(shape) * len(shape) * entropy)
        return train.Tensor(np.random.randint(0, 100, size=shape) / 20, dtype="float32")

    def get_int_tensor(self, *shape, low=0, high=10, entropy=1):
        np.random.seed(np.prod(shape) * len(shape) * entropy)
        return train.Tensor(np.random.randint(low, high, size=shape))

    def softmax_loss_forward(self, rows, classes):
        x = self.get_tensor(rows, classes)
        y = self.get_int_tensor(rows, low=0, high=classes)
        f = train.nn.SoftmaxLoss()
        return np.array(f(x, y).cached_data)

    def softmax_loss_backward(self, rows, classes):
        x = self.get_tensor(rows, classes)
        y = self.get_int_tensor(rows, low=0, high=classes)
        f = train.nn.SoftmaxLoss()
        loss = f(x, y)
        loss.backward()
        return x.grad.cached_data
    
    def test_nn_softmax_loss_forward_1(self):
        np.testing.assert_allclose(self.softmax_loss_forward(5, 10),
            np.array(4.041218, dtype=np.float32), rtol=1e-5, atol=1e-5)

    def test_nn_softmax_loss_forward_2(self):
        np.testing.assert_allclose(self.softmax_loss_forward(3, 11),
            np.array(3.3196716, dtype=np.float32), rtol=1e-5, atol=1e-5)

    def test_nn_softmax_loss_backward_1(self):
        np.testing.assert_allclose(self.softmax_loss_backward(5, 10),
            np.array([[ 0.00068890385, 0.0015331834 , 0.013162163 , -0.16422154 ,
            0.023983022 , 0.0050903494 , 0.00076135644, 0.050772052 ,
            0.0062173656 , 0.062013146 ],
            [ 0.012363418 , 0.02368262 , 0.11730081 , 0.001758993 ,
            0.004781439 , 0.0029000894 , -0.19815083 , 0.017544521 ,
            0.015874943 , 0.0019439887 ],
            [ 0.001219767 , 0.08134181 , 0.057320606 , 0.0008595553 ,
            0.0030001428 , 0.0009499555 , -0.19633561 , 0.0008176346 ,
            0.0014898272 , 0.0493363 ],
            [-0.19886842 , 0.08767337 , 0.017700946 , 0.026406704 ,
            0.0013147127 , 0.0107361665 , 0.009714483 , 0.023893777 ,
            0.019562569 , 0.0018656658 ],
            [ 0.007933789 , 0.017656967 , 0.027691642 , 0.0005605318 ,
            0.05576411 , 0.0013114461 , 0.06811045 , 0.011835824 ,
            0.0071787895 , -0.19804356 ]], dtype=np.float32), rtol=1e-5, atol=1e-5)

    def test_nn_softmax_loss_backward_2(self):
        np.testing.assert_allclose(self.softmax_loss_backward(3, 11),
            np.array([[ 0.0027466794, 0.020295369 , 0.012940894 , 0.04748398 ,
            0.052477922 , 0.090957515 , 0.0028875037, 0.012940894 ,
            0.040869843 , 0.04748398 , -0.33108455 ],
            [ 0.0063174255, 0.001721699 , 0.09400159 , 0.0034670753,
            0.038218185 , 0.009424488 , 0.0042346967, 0.08090791 ,
            -0.29697907 , 0.0044518122, 0.054234188 ],
            [ 0.14326698 , 0.002624026 , 0.0032049934, 0.01176007 ,
            0.045363605 , 0.0043262867, 0.039044812 , 0.017543964 ,
            0.0037236712, -0.3119051 , 0.04104668 ]], dtype=np.float32), rtol=1e-5, atol=1e-5)

if "__main__" == __name__:
    unittest.main()