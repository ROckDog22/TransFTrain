import unittest

import sys
sys.path.append('./python')
# 你需要在.vscode里面添加extra地址 才能找到
import TransFTrain as train
import numpy as np

op_conv_shapes = [
    ( (3, 14, 14, 8), (3, 3, 8, 16), 1, 0 ),
    ( (3, 14, 14, 8), (3, 3, 8, 16), 1, 1 ),
    ( (3, 16, 16, 8), (3, 3, 8, 16), 1, 2 ),
    ( (3, 16, 16, 8), (3, 3, 8, 14), 1, 0 ),
    ( (3, 16, 16, 2), (3, 3, 2, 14), 1, 0 ),

    ( (3, 14, 14, 8), (3, 3, 8, 16), 2, 0 ),
    ( (3, 14, 14, 8), (3, 3, 8, 16), 2, 1 ),
    ( (3, 16, 16, 8), (3, 3, 8, 16), 2, 2 ),
    ( (3, 16, 16, 8), (3, 3, 8, 14), 2, 0 ),
    ( (3, 16, 16, 2), (3, 3, 2, 14), 2, 0 ),

    ( (3, 16, 16, 24), (3, 3, 24, 14), 1, 0 ),
    ( (3, 14, 14, 8), (5, 5, 8, 16),   1, 0 ),
    ( (3, 17, 17, 8), (5, 5, 8, 16),   1, 0 ),
    ( (3, 17, 17, 1), (5, 5, 1, 16) ,  1, 0),
    ( (3, 17, 17, 16), (5, 5, 16, 1),  1, 0 ),
    ( (3, 17, 17, 16), (1, 1, 16, 1),  1, 0 ),
    ( (1, 14, 14, 2), (3, 3, 2, 2),    1, 0 ),
]

class TestDilate(unittest.TestCase):
    def test_op_conv_cpu(self):
        device = train.cpu()
        for Z_shape, W_shape, stride, padding in op_conv_shapes:
            for backward in [True, False]:
                np.random.seed(0)
                import torch
                _Z = np.random.randn(*Z_shape)*5
                _Z = _Z.astype(np.float32)
                _W = np.random.randn(*W_shape)*5
                _W = _W.astype(np.float32)
                Z = train.Tensor(_Z, device=device)
                W = train.Tensor(_W, device=device)
                y = train.conv(Z, W, padding=padding, stride=stride)
                y2 = y.sum()
                if backward:
                    y2.backward()
                Ztch = torch.Tensor(_Z).float()
                Ztch.requires_grad=True
                Wtch = torch.Tensor(_W).float()
                Wtch.requires_grad=True
                out = torch.nn.functional.conv2d(Ztch.permute(0, 3, 1, 2), Wtch.permute(3, 2, 0, 1), padding=padding, stride=stride)
                out2 = out.sum()
                if backward:
                    out2.backward()
                if backward:
                    err1 = np.linalg.norm(Ztch.grad.numpy() - Z.grad.numpy())
                    err2 = np.linalg.norm(Wtch.grad.numpy() - W.grad.numpy())
                err3 = np.linalg.norm(out2.detach().numpy() - y2.numpy())
                if backward:
                    assert err1 < 1e-2, "input grads match"
                    assert err2 < 1e-2, "weight grads match"
                assert err3 < 1e-1, "outputs match %s, %s" % (y2, out2)

    @unittest.skipIf(not train.cuda().enabled(), "NO GPU")
    def test_op_conv_cuda(self):
        device = train.cuda()
        for Z_shape, W_shape, stride, padding in op_conv_shapes:
            for backward in [True, False]:
                np.random.seed(0)
                import torch
                _Z = np.random.randn(*Z_shape)*5
                _Z = _Z.astype(np.float32)
                _W = np.random.randn(*W_shape)*5
                _W = _W.astype(np.float32)
                Z = train.Tensor(_Z, device=device)
                W = train.Tensor(_W, device=device)
                y = train.conv(Z, W, padding=padding, stride=stride)
                y2 = y.sum()
                if backward:
                    y2.backward()
                Ztch = torch.Tensor(_Z).float()
                Ztch.requires_grad=True
                Wtch = torch.Tensor(_W).float()
                Wtch.requires_grad=True
                out = torch.nn.functional.conv2d(Ztch.permute(0, 3, 1, 2), Wtch.permute(3, 2, 0, 1), padding=padding, stride=stride)
                out2 = out.sum()
                if backward:
                    out2.backward()
                if backward:
                    err1 = np.linalg.norm(Ztch.grad.numpy() - Z.grad.numpy())
                    err2 = np.linalg.norm(Wtch.grad.numpy() - W.grad.numpy())
                err3 = np.linalg.norm(out2.detach().numpy() - y2.numpy())
                if backward:
                    assert err1 < 1e-2, "input grads match"
                    assert err2 < 1e-2, "weight grads match"
                assert err3 < 1e-1, "outputs match %s, %s" % (y2, out2)

if __name__ == '__main__':
    unittest.main()