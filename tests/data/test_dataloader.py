import sys 
sys.path.append('./python')

import TransFTrain as train
import TransFTrain.nn as nn

import unittest
import numpy as np

class TestDataLoader(unittest.TestCase):  
    def test_dataloader_mnist():
        batch_size = 1
        mnist_train_dataset = train.data.MNISTDataset("data/train-images-idx3-ubyte.gz",
                                                    "data/train-labels-idx1-ubyte.gz")
        mnist_train_dataloader = train.data.DataLoader(dataset=mnist_train_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=False)

        for i, batch in enumerate(mnist_train_dataloader):
            batch_x, batch_y = batch[0].numpy(), batch[1].numpy()
            truth = mnist_train_dataset[i * batch_size:(i + 1) * batch_size]
            truth_x = truth[0] if truth[0].shape[0] > 1 else truth[0].reshape(-1)
            truth_y = truth[1] if truth[1].shape[0] > 1 else truth[1].reshape(-1)

            np.testing.assert_allclose(truth_x, batch_x.flatten())
            np.testing.assert_allclose(batch_y, truth_y)

        batch_size = 5
        mnist_test_dataset = train.data.MNISTDataset("data/t10k-images-idx3-ubyte.gz",
                                                "data/t10k-labels-idx1-ubyte.gz")
        mnist_test_dataloader = train.data.DataLoader(dataset=mnist_test_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=False)

        for i, batch in enumerate(mnist_test_dataloader):
            batch_x, batch_y = batch[0].numpy(), batch[1].numpy()
            truth = mnist_test_dataset[i * batch_size:(i + 1) * batch_size]
            truth_x = truth[0]
            truth_y = truth[1]

            np.testing.assert_allclose(truth_x, batch_x)
            np.testing.assert_allclose(batch_y, truth_y)



        noshuf = bat9 = train.data.DataLoader(dataset=mnist_test_dataset,
                                            batch_size=10,
                                            shuffle=False)
        shuf = bat9 = train.data.DataLoader(dataset=mnist_test_dataset,
                                        batch_size=10,
                                        shuffle=True)
        diff = False
        for i, j in zip(shuf, noshuf):
            if i != j:
                diff = True
                break
        assert diff, 'shuffling had no effect on the dataloader.'


    def test_dataloader_ndarray():
        for batch_size in [1,10,100]:
        np.random.seed(0)
        
        train_dataset = train.data.NDArrayDataset(np.random.rand(100,10,10))
        train_dataloader = train.data.DataLoader(dataset=train_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=False)

        for i, batch in enumerate(train_dataloader):
            batch_x  = batch[0].numpy()
            truth_x = train_dataset[i * batch_size:(i + 1) * batch_size][0].reshape((batch_size,10,10))
            np.testing.assert_allclose(truth_x, batch_x)

        batch_size = 1
        np.random.seed(0)
        train_dataset = train.data.NDArrayDataset(np.arange(100,))
        train_dataloader = iter(train.data.DataLoader(dataset=train_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=True))

        elements = np.array([next(train_dataloader)[0].numpy().item() for _ in range(10)])
        np.testing.assert_allclose(elements, np.array([26, 86,  2, 55, 75, 93, 16, 73, 54, 95]))

        batch_size = 10
        train_dataset = train.data.NDArrayDataset(np.arange(100,))
        train_dataloader = iter(train.data.DataLoader(dataset=train_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=True))

        elements = np.array([np.linalg.norm(next(train_dataloader)[0].numpy()) for _ in range(10)])
        np.testing.assert_allclose(elements, np.array([164.805946, 173.43875 , 169.841102, 189.050258, 195.880065, 206.387984, 209.909504, 185.776748, 145.948621, 160.252925]))

if "__main__" == __name__:
    unittest.main()
            
