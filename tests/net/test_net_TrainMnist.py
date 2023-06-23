import sys 
sys.path.append('./python')
sys.path.append('./tests/net')
import TransFTrain as train
import TransFTrain.nn as nn

import unittest
import numpy as np
from mlp_resnet import *

class TestTrainMnist(unittest.TestCase): 
    def train_mnist_1(self, batch_size, epochs, optimizer, lr, weight_decay, hidden_dim):
        np.random.seed(1)
        out = train_mnist(batch_size, epochs, optimizer, lr, weight_decay, hidden_dim, data_dir="./data")
        return np.array(out)

    def test_mlp_train_mnist_1(self):
        np.testing.assert_allclose(self.train_mnist_1(250, 2, train.optim.SGD, 0.001, 0.01, 100),
            np.array([0.4875 , 1.462595, 0.3245 , 1.049429]), rtol=0.001, atol=0.001)
if "__main__" == __name__:
    unittest.main()