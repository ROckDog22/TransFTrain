

import sys 
sys.path.append('./python')

import TransFTrain as train
import TransFTrain.nn as nn

import unittest
import numpy as np

class TestDataset(unittest.TestCase):     
    def test_mnist_dataset():
        # Test dataset sizing
        mnist_train_dataset = train.data.MNISTDataset("data/train-images-idx3-ubyte.gz",
                                                    "data/train-labels-idx1-ubyte.gz")
        assert len(mnist_train_dataset) == 60000

        sample_norms = np.array([np.linalg.norm(mnist_train_dataset[idx][0]) for idx in [1,42,1000,2000,3000,4000,5000,5005]])
        compare_against = np.array([10.188792, 6.261355, 8.966858, 9.4346485, 9.086626, 9.214664, 10.208544, 10.649756])
        sample_labels = np.array([mnist_train_dataset[idx][1] for idx in [1,42,1000,2000,3000,4000,5000,5005]])
        compare_labels = np.array([0,7,0,5,9,7,7,8])

        np.testing.assert_allclose(sample_norms, compare_against)
        np.testing.assert_allclose(sample_labels, compare_labels)

        mnist_train_dataset = train.data.MNISTDataset("data/t10k-images-idx3-ubyte.gz",
                                                "data/t10k-labels-idx1-ubyte.gz")
        assert len(mnist_train_dataset) == 10000

        sample_norms = np.array([np.linalg.norm(mnist_train_dataset[idx][0]) for idx in [1,42,1000,2000,3000,4000,5000,5005]])
        compare_against = np.array([9.857545, 8.980832, 8.57207 , 6.891522, 8.192135, 9.400087, 8.645003, 7.405202])
        sample_labels = np.array([mnist_train_dataset[idx][1] for idx in [1,42,1000,2000,3000,4000,5000,5005]])
        compare_labels = np.array([2, 4, 9, 6, 6, 9, 3, 1])

        np.testing.assert_allclose(sample_norms, compare_against, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(sample_labels, compare_labels)

        # test a transform
        np.random.seed(0)
        tforms = [train.data.RandomCrop(28), train.data.RandomFlipHorizontal()]
        mnist_train_dataset = train.data.MNISTDataset("data/train-images-idx3-ubyte.gz",
                                                    "data/train-labels-idx1-ubyte.gz",
                                                    transforms=tforms)

        sample_norms = np.array([np.linalg.norm(mnist_train_dataset[idx][0]) for idx in [1,42,1000,2000,3000,4000,5000,5005]])
        compare_against = np.array([2.0228338 ,0.        ,7.4892044 ,0.,0.,3.8012788,9.583429,4.2152724])
        sample_labels = np.array([mnist_train_dataset[idx][1] for idx in [1,42,1000,2000,3000,4000,5000,5005]])
        compare_labels = np.array([0,7,0,5,9,7,7,8])

        np.testing.assert_allclose(sample_norms, compare_against)
        np.testing.assert_allclose(sample_labels, compare_labels)


        # test a transform
        tforms = [train.data.RandomCrop(12), train.data.RandomFlipHorizontal(0.4)]
        mnist_train_dataset = train.data.MNISTDataset("data/train-images-idx3-ubyte.gz",
                                                    "data/train-labels-idx1-ubyte.gz",
                                                    transforms=tforms)
        sample_norms = np.array([np.linalg.norm(mnist_train_dataset[idx][0]) for idx in [1,42,1000,2000,3000,4000,5000,5005]])
        compare_against = np.array([5.369537, 5.5454974, 8.966858, 7.547235, 8.785921, 7.848442, 7.1654058, 9.361828])
        sample_labels = np.array([mnist_train_dataset[idx][1] for idx in [1,42,1000,2000,3000,4000,5000,5005]])
        compare_labels = np.array([0,7,0,5,9,7,7,8])

        np.testing.assert_allclose(sample_norms, compare_against, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(sample_labels, compare_labels)

if "__main__" == __name__:
    unittest.main()