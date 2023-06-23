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

    def mlp_resnet_num_params(self, dim, hidden_dim, num_blocks, num_classes, norm):
        model = MLPResNet(dim, hidden_dim, num_blocks, num_classes, norm)
        return np.array(self.num_params(model))

    def mlp_resnet_forward(self, dim, hidden_dim, num_blocks, num_classes, norm, drop_prob):
        np.random.seed(4)
        input_tensor = train.Tensor(np.random.randn(2, dim), dtype=np.float32)
        output_tensor = MLPResNet(dim, hidden_dim, num_blocks, num_classes, norm, drop_prob)(input_tensor)
        return output_tensor.numpy()

    def test_mlp_resnet_num_params_1(self):
        np.testing.assert_allclose(self.mlp_resnet_num_params(150, 100, 5, 10, nn.LayerNorm1d),
            np.array(68360), rtol=1e-5, atol=1e-5)

    def test_mlp_resnet_num_params_2(self):
        np.testing.assert_allclose(self.mlp_resnet_num_params(10, 100, 1, 100, nn.BatchNorm1d),
            np.array(21650), rtol=1e-5, atol=1e-5)

    def test_mlp_resnet_forward_1(self):
        np.testing.assert_allclose(
            self.mlp_resnet_forward(10, 5, 2, 5, nn.LayerNorm1d, 0.5),
            np.array([[3.046162, 1.44972, -1.921363, 0.021816, -0.433953],
                    [3.489114, 1.820994, -2.111306, 0.226388, -1.029428]],
                    dtype=np.float32),
            rtol=1e-5,
            atol=1e-5)

    def test_mlp_resnet_forward_2(self):
        np.testing.assert_allclose(
            self.mlp_resnet_forward(15, 25, 5, 14, nn.BatchNorm1d, 0.0),
            np.array([[
                0.92448235, -2.745743, -1.5077105, 1.130784, -1.2078242,
                -0.09833566, -0.69301605, 2.8945382, 1.259397, 0.13866742,
                -2.963875, -4.8566914, 1.7062538, -4.846424
            ],
            [
                0.6653336, -2.4708004, 2.0572243, -1.0791507, 4.3489094,
                3.1086435, 0.0304327, -1.9227124, -1.416201, -7.2151937,
                -1.4858506, 7.1039696, -2.1589825, -0.7593413
            ]],
            dtype=np.float32),
            rtol=1e-5,
            atol=1e-5)
        
if "__main__" == __name__:
    unittest.main()