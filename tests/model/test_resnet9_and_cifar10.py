import unittest

import sys
sys.path.append('./python')
sys.path.append('./tests')
# 你需要在.vscode里面添加extra地址 才能找到
import TransFTrain as train
import numpy as np
from model.models import ResNet9
def one_iter_of_cifar10_training(dataloader, model, niter=1, loss_fn=train.nn.SoftmaxLoss(), opt=None, device=None):
    np.random.seed(4)
    model.train()
    correct, total_loss = 0, 0
    i = 1
    for batch in dataloader:
        opt.reset_grad()
        X, y = batch
        X,y = train.Tensor(X, device=device), train.Tensor(y, device=device)
        out = model(X)
        correct += np.sum(np.argmax(out.numpy(), axis=1) == y.numpy())
        loss = loss_fn(out, y)
        total_loss += loss.data.numpy() * y.shape[0]
        loss.backward()
        opt.step()
        if i >= niter:
            break
        i += 1
    return correct/(y.shape[0]*niter), total_loss/(y.shape[0]*niter)

class TestResNet9andCifar10(unittest.TestCase):
    def test_resnet9_cpu(self):
        device = train.cpu()
        def num_params(model):
            return np.sum([np.prod(x.shape) for x in model.parameters()])
        np.random.seed(0)
        model = ResNet9(device=device)

        assert num_params(model) == 431946

        _A = np.random.randn(2, 3, 32, 32)
        A = train.Tensor(_A, device=device)
        y = model(A)

        assert np.linalg.norm(np.array([[-1.8912625 ,  0.64833605,  1.9400386 ,  1.1435282 ,  1.89777   ,
            2.9039745 , -0.10433993,  0.35458302, -0.5684191 ,  2.6178317 ],
        [-0.2905612 , -0.4147861 ,  0.90268034,  0.46530387,  1.3335679 ,
            1.8534894 , -0.1867125 , -2.4298222 , -0.5344223 ,  4.362149  ]]) - y.numpy()) < 1e-2

    @unittest.skipIf(not train.cuda().enabled(), "NO GPU")
    def test_resnet9_cuda(self):
        device = train.cuda()
        def num_params(model):
            return np.sum([np.prod(x.shape) for x in model.parameters()])
        np.random.seed(0)
        model = ResNet9(device=device)

        assert num_params(model) == 431946

        _A = np.random.randn(2, 3, 32, 32)
        A = train.Tensor(_A, device=device)
        y = model(A)

        assert np.linalg.norm(np.array([[-1.8912625 ,  0.64833605,  1.9400386 ,  1.1435282 ,  1.89777   ,
            2.9039745 , -0.10433993,  0.35458302, -0.5684191 ,  2.6178317 ],
        [-0.2905612 , -0.4147861 ,  0.90268034,  0.46530387,  1.3335679 ,
            1.8534894 , -0.1867125 , -2.4298222 , -0.5344223 ,  4.362149  ]]) - y.numpy()) < 1e-2
        
    def test_train_cifar10_cpu(self):
        device = train.cpu()
        np.random.seed(0)
        dataset = train.data.CIFAR10Dataset("./data/cifar-10-batches-py", train=True)
        dataloader = train.data.DataLoader(\
                dataset=dataset,
                batch_size=128,
                shuffle=False
                # collate_fn=train.data.collate_ndarray,
                # drop_last=False,
                # device=device,
                # dtype="float32"
                )
        np.random.seed(0)
        model = ResNet9(device=device, dtype="float32")
        out = one_iter_of_cifar10_training(dataloader, model, opt=train.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001), device=device)
        assert np.linalg.norm(np.array(list(out)) - np.array([0.09375, 3.5892258])) < 1e-2


    @unittest.skipIf(not train.cuda().enabled(), "NO GPU")
    def test_train_cifar10_cuda(self):
        device = train.cuda()
        np.random.seed(0)
        dataset = train.data.CIFAR10Dataset("./data/cifar-10-batches-py", train=True)
        dataloader = train.data.DataLoader(\
                dataset=dataset,
                batch_size=128,
                shuffle=False
                # collate_fn=train.data.collate_ndarray,
                # drop_last=False,
                # device=device,
                # dtype="float32"
                )
        np.random.seed(0)
        model = ResNet9(device=device, dtype="float32")
        out = one_iter_of_cifar10_training(dataloader, model, opt=train.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001), device=device)
        assert np.linalg.norm(np.array(list(out)) - np.array([0.09375, 3.5892258])) < 1e-2

if __name__ == '__main__':
    unittest.main()