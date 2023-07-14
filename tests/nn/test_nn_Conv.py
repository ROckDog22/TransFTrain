import unittest

import sys
sys.path.append('./python')
# 你需要在.vscode里面添加extra地址 才能找到
import TransFTrain as train
import numpy as np

conv_forward_params = [
    (4, 8, 16, 3, 1),
    (32, 8, 16, 3, 2),
    (32, 8, 8, 3, 2),
    (32, 16, 8, 3, 1),
    (32, 16, 8, 3, 2)
]

conv_back_params = [
    (4, 1, 1, 3, 1),
    (14, 8, 16, 3, 1),
    (14, 8, 16, 3, 2),
    (14, 8, 8, 3, 1),
    (14, 8, 8, 3, 2),
    (14, 16, 8, 3, 1),
    (14, 16, 8, 3, 2),
]


class TestNNConv(unittest.TestCase):
    def test_nn_conv_forward_cpu(self):
        device = train.cpu()
        for s,cin, cout, k, stride in conv_forward_params:
            np.random.seed(0)
            import torch
            f = train.nn.Conv(cin, cout, k, stride=stride, device=device)
            x = train.init.rand(10, cin, s, s, device=device)

            g = torch.nn.Conv2d(cin, cout, k, stride=stride, padding=k//2)
            g.weight.data = torch.tensor(f.weight.cached_data.numpy().transpose(3, 2, 0, 1))
            g.bias.data = torch.tensor(f.bias.cached_data.numpy())
            z = torch.tensor(x.cached_data.numpy())

            assert np.linalg.norm(f(x).cached_data.numpy() - g(z).data.numpy()) < 1e-3

    @unittest.skipIf(not train.cuda().enabled(), "NO GPU")
    def test_nn_conv_forward_cuda(self):
        device = train.cuda()
        for s,cin, cout, k, stride in conv_forward_params:
            np.random.seed(0)
            import torch
            f = train.nn.Conv(cin, cout, k, stride=stride, device=device)
            x = train.init.rand(10, cin, s, s, device=device)

            g = torch.nn.Conv2d(cin, cout, k, stride=stride, padding=k//2)
            g.weight.data = torch.tensor(f.weight.cached_data.numpy().transpose(3, 2, 0, 1))
            g.bias.data = torch.tensor(f.bias.cached_data.numpy())
            z = torch.tensor(x.cached_data.numpy())

            assert np.linalg.norm(f(x).cached_data.numpy() - g(z).data.numpy()) < 1e-3


    def test_nn_conv_backward_cpu(self):
        device = train.cpu()
        for s,cin,cout,k,stride in conv_back_params:
            np.random.seed(0)
            import torch
            f = train.nn.Conv(cin, cout, k, stride=stride, device=device)
            x = train.init.rand(1, cin, s, s, device=device, requires_grad=True)

            g = torch.nn.Conv2d(cin, cout, k, stride=stride, padding=k//2)
            g.weight.data = torch.tensor(f.weight.cached_data.numpy().transpose(3, 2, 0, 1))
            g.bias.data = torch.tensor(f.bias.cached_data.numpy())
            z = torch.tensor(x.cached_data.numpy(), requires_grad=True)
            z.requires_grad = True

            res1 = f(x)
            y1 = res1.sum()

            y2 = g(z).sum()

            y1.backward()
            y2.backward()

            assert np.linalg.norm(g.weight.grad.data.numpy() - f.weight.grad.cached_data.numpy().transpose(3, 2, 0, 1)) < 1e-3, "weight gradients match"
            assert np.linalg.norm(g.bias.grad.data.numpy() - f.bias.grad.cached_data.numpy()) < 1e-3, "bias gradients match"
            assert np.linalg.norm(z.grad.data.numpy() - x.grad.cached_data.numpy()) < 1e-3, "input gradients match"

    @unittest.skipIf(not train.cuda().enabled(), "NO GPU")
    def test_nn_conv_backward_cuda(self):
        device = train.cpu()
        for s,cin,cout,k,stride in conv_back_params:
            np.random.seed(0)
            import torch
            f = train.nn.Conv(cin, cout, k, stride=stride, device=device)
            x = train.init.rand(1, cin, s, s, device=device, requires_grad=True)

            g = torch.nn.Conv2d(cin, cout, k, stride=stride, padding=k//2)
            g.weight.data = torch.tensor(f.weight.cached_data.numpy().transpose(3, 2, 0, 1))
            g.bias.data = torch.tensor(f.bias.cached_data.numpy())
            z = torch.tensor(x.cached_data.numpy(), requires_grad=True)
            z.requires_grad = True

            res1 = f(x)
            y1 = res1.sum()

            y2 = g(z).sum()

            y1.backward()
            y2.backward()

            assert np.linalg.norm(g.weight.grad.data.numpy() - f.weight.grad.cached_data.numpy().transpose(3, 2, 0, 1)) < 1e-3, "weight gradients match"
            assert np.linalg.norm(g.bias.grad.data.numpy() - f.bias.grad.cached_data.numpy()) < 1e-3, "bias gradients match"
            assert np.linalg.norm(z.grad.data.numpy() - x.grad.cached_data.numpy()) < 1e-3, "input gradients match"

if __name__ == '__main__':
    unittest.main()