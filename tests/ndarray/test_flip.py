import unittest

import sys
sys.path.append('./python')
import TransFTrain as train
import TransFTrain.backend_ndarray as nd
import numpy as np
import torch

flip_forward_params = [
    {"shape": (10, 5), "axes": (0,)},
    {"shape": (10, 5), "axes": (1,)},
    {"shape": (10, 5), "axes": (0,1)},
    {"shape": (10, 32, 32, 8), "axes": (0,1)},
    {"shape": (3, 3, 6, 8), "axes": (0,1)},
    {"shape": (10, 32, 32, 8), "axes": (1,2)},
    {"shape": (3, 3, 6, 8), "axes": (1,2)},
    {"shape": (10, 32, 32, 8), "axes": (2,3)},
    {"shape": (3, 3, 6, 8), "axes": (2,3)},
    {"shape": (10, 32, 32, 8), "axes": (0,1,2,3)},
]

flip_backward_params = [
    {"shape": (10, 5), "axes": (0,)},
    {"shape": (10, 5), "axes": (1,)},
    {"shape": (10, 5), "axes": (0,1)},
    {"shape": (2, 3, 3, 8), "axes": (0,1)},
    {"shape": (3, 3, 6, 4), "axes": (0,1)},
    {"shape": (2, 3, 3, 4), "axes": (1,2)},
    {"shape": (3, 3, 6, 4), "axes": (1,2)},
    {"shape": (2, 3, 3, 4), "axes": (2,3)},
    {"shape": (3, 3, 6, 4), "axes": (2,3)},
    {"shape": (2, 3, 3, 4), "axes": (0,1,2,3)},
]

def backward_check(f, *args, **kwargs):
    eps = 1e-3
    out = f(*args, **kwargs)
    c = np.random.randn(*out.shape)
    is_stacked = False
    if isinstance(args[0], list):
        args = args[0]
        is_stacked = True
    numerical_grad = [np.zeros(a.shape) for a in args]
    num_args = len(args)
    for i in range(num_args):
        for j in range(args[i].realize_cached_data().size):
            args[i].realize_cached_data().flat[j] += eps
            if is_stacked:
                f1 = (f(args, **kwargs).numpy() * c).sum()
            else:
                f1 = (f(*args, **kwargs).numpy() * c).sum()
            args[i].realize_cached_data().flat[j] -= 2 * eps
            if is_stacked:
                f2 = (f(args, **kwargs).numpy() * c).sum()
            else:
                f2 = (f(*args, **kwargs).numpy() * c).sum()
            args[i].realize_cached_data().flat[j] += eps
            numerical_grad[i].flat[j] = (f1 - f2) / (2 * eps)
    backward_grad = out.op.gradient_as_tuple(train.Tensor(c, device=args[0].device), out)
    if isinstance(backward_grad[0], train.TensorTuple): # TODO keep this?
        backward_grad = backward_grad[0].tuple()
    error = sum(
        np.linalg.norm(backward_grad[i].numpy() - numerical_grad[i])
        for i in range(len(args))
    )
    assert error < 1e-2
    return [g.numpy() for g in backward_grad]


class TestFlip(unittest.TestCase):
    def test_flip_forward_cpu(self):
        device = train.cpu()
        for params in flip_forward_params:
            np.random.seed(0)
            shape, axes = params['shape'], params['axes']
            _A = np.random.randn(*shape)
            _B = np.flip(_A, axes)
            A = train.Tensor(_A, device=device)
            B = train.flip(A, axes=axes)

            assert np.linalg.norm(B.numpy() - _B) < 1e-4


    def test_flip_forward_cuda(self):
        device = train.cuda()
        for params in flip_forward_params:
            np.random.seed(0)
            shape, axes = params['shape'], params['axes']
            _A = np.random.randn(*shape)
            _B = np.flip(_A, axes)
            A = train.Tensor(_A, device=device)
            B = train.flip(A, axes=axes)

            assert np.linalg.norm(B.numpy() - _B) < 1e-4

    def test_flip_backward_cpu(self):
        device = train.cpu()
        np.random.seed(0)
        for params in flip_backward_params:
            shape, axes = params['shape'], params['axes']
            backward_check(train.flip, train.Tensor(np.random.randn(*shape), device=device), axes=axes)

    def test_flip_backward_cuda(self):
        device = train.cuda()
        np.random.seed(0)
        for params in flip_backward_params:
            shape, axes = params['shape'], params['axes']
            backward_check(train.flip, train.Tensor(np.random.randn(*shape), device=device), axes=axes)


if __name__=="__main__":
    unittest.main()
