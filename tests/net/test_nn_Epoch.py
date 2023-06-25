import sys 
sys.path.append('./python')
sys.path.append('./tests/net')
import TransFTrain as train
import TransFTrain.nn as nn

import unittest
import numpy as np
from mlp_resnet import *

class TestResidualBlock(unittest.TestCase): 
    def train_epoch_1(self, hidden_dim, batch_size, optimizer, **kwargs):
        np.random.seed(1)
        train_dataset = train.data.MNISTDataset(\
                "./data/train-images-idx3-ubyte.gz",
                "./data/train-labels-idx1-ubyte.gz")
        train_dataloader = train.data.DataLoader(\
                dataset=train_dataset,
                batch_size=batch_size)

        model = MLPResNet(784, hidden_dim)
        opt = optimizer(model.parameters(), **kwargs)
        model.eval()
        return np.array(epoch(train_dataloader, model, opt))


    def test_mlp_train_epoch_1(self):
        np.testing.assert_allclose(self.train_epoch_1(5, 250, train.optim.Adam, lr=0.01, weight_decay=0.1),
            np.array([0.675267, 1.84043]), rtol=0.0001, atol=0.0001)

    def test_mlp_eval_epoch_1(self):
        np.testing.assert_allclose(self.eval_epoch_1(10, 150),
            np.array([0.9164 , 4.137814]), rtol=1e-5, atol=1e-5)


    def eval_epoch_1(self, hidden_dim, batch_size):
        np.random.seed(1)
        test_dataset = train.data.MNISTDataset(\
                "./data/t10k-images-idx3-ubyte.gz",
                "./data/t10k-labels-idx1-ubyte.gz")
        test_dataloader = train.data.DataLoader(\
                dataset=test_dataset,
                batch_size=batch_size,
                shuffle=False)

        model = MLPResNet(784, hidden_dim)
        model.train()
        return np.array(epoch(test_dataloader, model))

if "__main__" == __name__:
    unittest.main()
#     suite = unittest.TestSuite()
#     suite.addTest(TestResidualBlock("test_mlp_eval_epoch_1"))
#     runner = unittest.TextTestRunner()
#     result = runner.run(suite)