import unittest

import sys
sys.path.append('./python')
import TransFTrain as train
import TransFTrain.backend_ndarray as nd
import numpy as np
import torch

pad_params = [
    {"shape": (10, 32, 32, 8), "padding": ( (0, 0), (2, 2), (2, 2), (0, 0) )},
    {"shape": (10, 32, 32, 8), "padding": ( (0, 0), (0, 0), (0, 0), (0, 0) )},
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


class TestPad(unittest.TestCase):
    def test_pad_forward_cpu(self):
        device = train.cpu()
        for params in pad_params:
            np.random.seed(0)
            shape, padding = params['shape'], params['padding']
            _A = np.random.randn(*shape)
            _B = np.pad(_A, padding)
            A = nd.NDArray(_A, device=device)
            B = A.pad(padding)

            assert np.linalg.norm(B.numpy() - _B) < 1e-4


    def test_pad_forward_cuda(self):
        device = train.cuda()
        for params in pad_params:
            np.random.seed(0)
            shape, padding = params['shape'], params['padding']
            _A = np.random.randn(*shape)
            _B = np.pad(_A, padding)
            A = nd.NDArray(_A, device=device)
            B = A.pad(padding)

            assert np.linalg.norm(B.numpy() - _B) < 1e-4

            
if __name__=="__main__":
    unittest.main()
