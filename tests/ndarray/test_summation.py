import unittest

import sys
sys.path.append('./python')
import TransFTrain as train
import TransFTrain.backend_ndarray as nd
import numpy as np
import torch

SUMMATION_PARAMETERS = [((1, 1, 1), None),
    ((5, 3), 0),
    ((8, 3, 2), 1),
    ((8, 3, 2), 2)
]

np.random.seed(1)

def backward_check(f, *args, **kwargs):
    eps = 1e-5
    out = f(*args, **kwargs)
    c = np.random.randn(*out.shape)
    numerical_grad = [np.zeros(a.shape) for a in args]
    num_args = len(args)
    for i in range(num_args):
        for j in range(args[i].realize_cached_data().size):
            args[i].realize_cached_data().flat[j] += eps
            f1 = (f(*args, **kwargs).numpy() * c).sum()
            args[i].realize_cached_data().flat[j] -= 2 * eps
            f2 = (f(*args, **kwargs).numpy() * c).sum()
            args[i].realize_cached_data().flat[j] += eps
            numerical_grad[i].flat[j] = (f1 - f2) / (2 * eps)
    backward_grad = out.op.gradient_as_tuple(train.Tensor(c, device=args[0].device), out)
    error = sum(
        np.linalg.norm(backward_grad[i].numpy() - numerical_grad[i])
        for i in range(len(args))
    )
    assert error < 4.2e-1
    return [g.numpy() for g in backward_grad]

class TestStack(unittest.TestCase):
    def test_summation_cpu(self):
        device = train.cpu()
        for shape, axes in SUMMATION_PARAMETERS:
            _A = np.random.randn(*shape).astype(np.float32)
            A = train.Tensor(nd.array(_A), device=device)
            np.testing.assert_allclose(np.sum(_A, axes), train.summation(A, axes=axes).numpy(), atol=1e-5, rtol=1e-5)

    @unittest.skipIf(not nd.cuda().enabled(), "NO GPU")
    def test_summation_cuda(self):
        device = train.cuda()
        for shape, axes in SUMMATION_PARAMETERS:
            _A = np.random.randn(*shape).astype(np.float32)
            A = train.Tensor(nd.array(_A), device=device)
            np.testing.assert_allclose(np.sum(_A, axes), train.summation(A, axes=axes).numpy(), atol=1e-5, rtol=1e-5)

    def test_summation_backward_cpu(self):
        device = train.cpu()
        for shape, axes in SUMMATION_PARAMETERS:
            _A = np.random.randn(*shape).astype(np.float32)
            A = train.Tensor(nd.array(_A), device=device)
            backward_check(train.summation, A, axes=axes)

    @unittest.skipIf(not nd.cuda().enabled(), "NO GPU")
    def test_summation_backward_cuda(self):
        device = train.cuda()
        for shape, axes in SUMMATION_PARAMETERS:
            _A = np.random.randn(*shape).astype(np.float32)
            A = train.Tensor(nd.array(_A), device=device)
            backward_check(train.summation, A, axes=axes)

if __name__=="__main__":
    unittest.main()