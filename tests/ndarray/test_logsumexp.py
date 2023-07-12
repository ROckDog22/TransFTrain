import unittest

import sys
sys.path.append('./python')
# 你需要在.vscode里面添加extra地址 才能找到
import TransFTrain as train
import TransFTrain.backend_ndarray as nd
import numpy as np
import torch

SUMMATION_PARAMETERS = [((1, 1, 1), None),
    ((5, 3), 0),
    ((8, 3, 2), 1),
    ((8, 3, 2), 2)
]
class TestEwiseLog(unittest.TestCase):
    def test_logsumexp_cpu(self):
        device = train.cpu()
        for shape, axes in SUMMATION_PARAMETERS:
            _A = np.random.randn(*shape).astype(np.float32)
            A = train.Tensor(nd.array(_A), device=device)
            A_t = torch.Tensor(_A)
            if axes is None:
                t_axes = tuple(list(range(len(shape))))
            else:
                t_axes = axes
            np.testing.assert_allclose(torch.logsumexp(A_t, dim=t_axes).numpy(), train.logsumexp(A, axes=axes).numpy(), atol=1e-5, rtol=1e-5)

    def test_logsumexp_cuda(self):
        device = train.cuda()
        for shape, axes in SUMMATION_PARAMETERS:
            _A = np.random.randn(*shape).astype(np.float32)
            A = train.Tensor(nd.array(_A), device=device)
            A_t = torch.Tensor(_A)
            if axes is None:
                t_axes = tuple(list(range(len(shape))))
            else:
                t_axes = axes
            np.testing.assert_allclose(torch.logsumexp(A_t, dim=t_axes).numpy(), train.logsumexp(A, axes=axes).numpy(), atol=1e-5, rtol=1e-5)


if "__main__" == __name__:
    unittest.main()