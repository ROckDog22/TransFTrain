import sys 
sys.path.append('./python')
sys.path.append('./tests/net')
import TransFTrain as train
import TransFTrain.nn as nn

import unittest
import numpy as np
from mlp_resnet import *

class TestResidualBlock(unittest.TestCase): 
    def get_tensor(self, *shape, entropy=1):
        np.random.seed(np.prod(shape) * len(shape) * entropy)
        return train.Tensor(np.random.randint(0, 100, size=shape) / 20, dtype="float32")

    def get_int_tensor(self, *shape, low=0, high=10, entropy=1):
        np.random.seed(np.prod(shape) * len(shape) * entropy)
        return train.Tensor(np.random.randint(low, high, size=shape))

    def num_params(self, model):
        return np.sum([np.prod(x.shape) for x in model.parameters()])

    def residual_block_num_params(self, dim, hidden_dim, norm):
        model = ResidualBlock(dim, hidden_dim, norm)
        return np.array(self.num_params(model))

    def residual_block_forward(self, dim, hidden_dim, norm, drop_prob):
        np.random.seed(2)
        input_tensor = train.Tensor(np.random.randn(1, dim))
        output_tensor = ResidualBlock(dim, hidden_dim, norm, drop_prob)(input_tensor)
        return output_tensor.numpy()
    
    def test_mlp_residual_block_num_params_1(self):
        np.testing.assert_allclose(self.residual_block_num_params(15, 2, nn.BatchNorm1d),
            np.array(111), rtol=1e-5, atol=1e-5)

    def test_mlp_residual_block_num_params_2(self):
        np.testing.assert_allclose(self.residual_block_num_params(784, 100, nn.LayerNorm1d),
            np.array(159452), rtol=1e-5, atol=1e-5)

    def test_mlp_residual_block_forward_1(self):
        np.testing.assert_allclose(
            self.residual_block_forward(15, 10, nn.LayerNorm1d, 0.5),
            np.array([[
                0., 1.358399, 0., 1.384224, 0., 0., 0.255451, 0.077662, 0.,
                0.939582, 0.525591, 1.99213, 0., 0., 1.012827
            ]],
            dtype=np.float32),
            rtol=1e-5,
            atol=1e-5,
        )

if "__main__" == __name__:
    unittest.main()
    # suite = unittest.TestSuite()
    # suite.addTest(TestResidualBlock("test_mlp_residual_block_num_params_1"))
    # runner = unittest.TextTestRunner()
    # result = runner.run(suite)