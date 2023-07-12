import sys 
sys.path.append('./python')

import TransFTrain as train
import TransFTrain.nn as nn
import TransFTrain.backend_ndarray as nd

import unittest
import numpy as np

np.random.seed(2)


TRAIN = [True, False]
BATCH_SIZES = [1, 15]
BPTT = [3, 32]

class TestCifarPtbData(unittest.TestCase):
    def test_cifar10_dataset(self):
        for train_ in TRAIN:
            dataset = train.data.CIFAR10Dataset("data/cifar-10-batches-py", train=train_)
            if train_:
                assert len(dataset) == 50000
            else:
                assert len(dataset) == 10000
            example = dataset[np.random.randint(len(dataset))]
            assert(isinstance(example, tuple))
            X, y = example
            assert isinstance(X, np.ndarray)
            assert X.shape == (3, 32, 32)

    def test_cifar10_loader(self):
        for batch_size in BATCH_SIZES:
            cifar10_train_dataset = train.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
            train_loader = train.data.DataLoader(cifar10_train_dataset, batch_size)
            for (X, y) in train_loader:
                break
            assert isinstance(X.cached_data, nd.NDArray)
            assert isinstance(X, train.Tensor)
            assert isinstance(y, train.Tensor)
            assert X.dtype == 'float32'

    def test_ptb_dataset_cpu(self):
        device = nd.cpu()
        for batch_size in BATCH_SIZES:
            for bptt in BPTT:
                for train_ in TRAIN:
                    corpus = train.data.Corpus("data/ptb")
                    if train_:
                        data = train.data.batchify(corpus.train, batch_size, device=device, dtype="float32")
                    else:
                        data = train.data.batchify(corpus.test, batch_size, device=device, dtype="float32")
                    X, y = train.data.get_batch(data, np.random.randint(len(data)), bptt, device=device)
                    assert X.shape == (bptt, batch_size)
                    assert y.shape == (bptt * batch_size,)
                    assert isinstance(X, train.Tensor)
                    assert X.dtype == 'float32'
                    assert X.device == device
                    assert isinstance(X.cached_data, nd.NDArray)
                    ntokens = len(corpus.dictionary)
                    assert ntokens == 10000

    @unittest.skipIf(not nd.cuda().enabled(), "NO GPU")
    def test_ptb_dataset_cuda(self):
        device = nd.cuda()
        for batch_size in BATCH_SIZES:
            for bptt in BPTT:
                for train_ in TRAIN:
                    corpus = train.data.Corpus("data/ptb")
                    if train_:
                        data = train.data.batchify(corpus.train, batch_size, device=device, dtype="float32")
                    else:
                        data = train.data.batchify(corpus.test, batch_size, device=device, dtype="float32")
                    X, y = train.data.get_batch(data, np.random.randint(len(data)), bptt, device=device)
                    assert X.shape == (bptt, batch_size)
                    assert y.shape == (bptt * batch_size,)
                    assert isinstance(X, train.Tensor)
                    assert X.dtype == 'float32'
                    assert X.device == device
                    assert isinstance(X.cached_data, nd.NDArray)
                    ntokens = len(corpus.dictionary)
                    assert ntokens == 10000

if "__main__" == __name__:
    unittest.main()
