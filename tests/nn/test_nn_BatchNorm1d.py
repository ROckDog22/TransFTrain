import sys 
sys.path.append('./python')

import TransFTrain as train
import TransFTrain.nn as nn

import unittest
import numpy as np

class UselessModule(train.nn.Module):
    def __init__(self):
        super().__init__()
        self.stuff = {'layer1': nn.Linear(4, 4),
                    'layer2': [nn.Dropout(0.1), nn.Sequential(nn.Linear(4, 4))]}

    def forward(self, x):
        raise NotImplementedError()
    
class TestBatchNorm1d(unittest.TestCase): 
    def get_tensor(self, *shape, entropy=1):
        np.random.seed(np.prod(shape) * len(shape) * entropy)
        return train.Tensor(np.random.randint(0, 100, size=shape) / 20, dtype="float32")

    def get_int_tensor(self, *shape, low=0, high=10, entropy=1):
        np.random.seed(np.prod(shape) * len(shape) * entropy)
        return train.Tensor(np.random.randint(low, high, size=shape))
        
    def check_training_mode(self):
        model = nn.Sequential(
            nn.BatchNorm1d(4),
            nn.Sequential(
                nn.LayerNorm1d(4),
                nn.Linear(4, 4),
                nn.Dropout(0.1),
            ),
            nn.Linear(4, 4),
            UselessModule()
        )

        model_refs = [
            model.modules[0],
            model.modules[1].modules[0],
            model.modules[1].modules[1],
            model.modules[1].modules[2],
            model.modules[2],
            model.modules[3],
            model.modules[3].stuff['layer1'],
            model.modules[3].stuff['layer2'][0],
            model.modules[3].stuff['layer2'][1].modules[0]
        ]

        eval_mode = [1 if not x.training else 0 for x in model_refs]
        model.eval()
        eval_mode.extend([1 if not x.training else 0 for x in model_refs])
        model.train()
        eval_mode.extend([1 if not x.training else 0 for x in model_refs])

        return np.array(eval_mode)
    
    def batchnorm_forward(self, *shape, affine=False):
        x = self.get_tensor(*shape)
        bn = train.nn.BatchNorm1d(shape[1])
        if affine:
            bn.weight.data = self.get_tensor(shape[1], entropy=42)
            bn.bias.data = self.get_tensor(shape[1], entropy=1337)
        return bn(x).cached_data

    def batchnorm_backward(self, *shape, affine=False):
        x = self.get_tensor(*shape)
        bn = train.nn.BatchNorm1d(shape[1])
        if affine:
            bn.weight.data = self.get_tensor(shape[1], entropy=42)
            bn.bias.data = self.get_tensor(shape[1], entropy=1337)
        y = (bn(x)**2).sum().backward()
        return x.grad.cached_data

    def batchnorm_running_mean(self, *shape, iters=10):
        bn = train.nn.BatchNorm1d(shape[1])
        for i in range(iters):
            x = self.get_tensor(*shape, entropy=i)
            y = bn(x)
        return bn.running_mean.cached_data

    def batchnorm_running_var(self, *shape, iters=10):
        bn = train.nn.BatchNorm1d(shape[1])
        for i in range(iters):
            x = self.get_tensor(*shape, entropy=i)
            y = bn(x)
        return bn.running_var.cached_data

    def batchnorm_running_grad(self, *shape, iters=10):
        bn = train.nn.BatchNorm1d(shape[1])
        for i in range(iters):
            x = self.get_tensor(*shape, entropy=i)
            y = bn(x)
        bn.eval()
        (y**2).sum().backward()
        return x.grad.cached_data

    def test_nn_batchnorm_check_model_eval_switches_training_flag_1(self):
        np.testing.assert_allclose(self.check_training_mode(),
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
            0, 0, 0, 0, 0]), rtol=1e-5, atol=1e-5)

    def test_nn_batchnorm_forward_1(self):
        np.testing.assert_allclose(self.batchnorm_forward(4, 4),
            np.array([[ 7.8712696e-01, -3.1676728e-01, -6.4885163e-01, 2.0828949e-01],
            [-7.9508079e-03, 1.0092355e+00, 1.6221288e+00, 8.5209310e-01],
            [ 8.5073310e-01, -1.4954363e+00, -9.6686421e-08, -1.6852506e+00],
            [-1.6299094e+00, 8.0296844e-01, -9.7327745e-01, 6.2486827e-01]],
            dtype=np.float32), rtol=1e-5, atol=1e-5)



    def test_nn_batchnorm_forward_affine_1(self):
        np.testing.assert_allclose(self.batchnorm_forward(4, 4, affine=True),
            np.array([[ 7.49529 , 0.047213316, 2.690084 , 5.5227957 ],
            [ 4.116209 , 3.8263211 , 7.79979 , 7.293256 ],
            [ 7.765616 , -3.3119934 , 4.15 , 0.31556034 ],
            [-2.7771149 , 3.23846 , 1.9601259 , 6.6683874 ]],
            dtype=np.float32), rtol=1e-5, atol=1e-5)


    def test_nn_batchnorm_backward_1(self):
        np.testing.assert_allclose(self.batchnorm_backward(5, 4),
            np.array([[ 2.1338463e-04, 5.2094460e-06, -2.8359889e-05, -4.4368207e-06],
            [-3.8480759e-04, -4.0292739e-06, 1.8370152e-05, -1.1172146e-05],
            [ 2.5629997e-04, -1.1003018e-05, -9.0479853e-06, 5.5171549e-06],
            [-4.2676926e-04, 3.4213067e-06, 1.3601780e-05, 1.0166317e-05],
            [ 3.4189224e-04, 6.4015389e-06, 5.4359434e-06, -7.4505806e-08]],
            dtype=np.float32), rtol=1e-5, atol=1e-5)

    def test_nn_batchnorm_backward_affine_1(self):
        np.testing.assert_allclose(self.batchnorm_backward(5, 4, affine=True),
            np.array([[ 3.8604736e-03, 4.2676926e-05, -1.4114380e-04, -3.2424927e-05],
            [-6.9427490e-03, -3.3140182e-05, 9.1552734e-05, -8.5830688e-05],
            [ 4.6386719e-03, -8.9883804e-05, -4.5776367e-05, 4.3869019e-05],
            [-7.7133179e-03, 2.7418137e-05, 6.6757202e-05, 7.4386597e-05],
            [ 6.1874390e-03, 5.2213669e-05, 2.8610229e-05, -1.9073486e-06]],
            dtype=np.float32), rtol=1e-5, atol=1e-4)


    def test_nn_batchnorm_running_mean_1(self):
        np.testing.assert_allclose(self.batchnorm_running_mean(4, 3),
            np.array([2.020656, 1.69489 , 1.498846], dtype=np.float32), rtol=1e-5, atol=1e-5)



    def test_nn_batchnorm_running_var_1(self):
        np.testing.assert_allclose(self.batchnorm_running_var(4, 3),
            np.array([1.412775, 1.386191, 1.096604], dtype=np.float32), rtol=1e-5, atol=1e-5)



    def test_nn_batchnorm_running_grad_1(self):
        np.testing.assert_allclose(self.batchnorm_running_grad(4, 3),
            np.array([[ 8.7022781e-06, -4.9751252e-06, 9.5367432e-05],
            [ 6.5565109e-06, -7.2401017e-06, -2.3484230e-05],
            [-3.5762787e-06, -4.5262277e-07, 1.6093254e-05],
            [-1.1682510e-05, 1.2667850e-05, -8.7976456e-05]], dtype=np.float32), rtol=1e-5, atol=1e-5)

if "__main__" == __name__:
    unittest.main()
