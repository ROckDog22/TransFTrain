import unittest

import sys
sys.path.append('./python')
# 你需要在.vscode里面添加extra地址 才能找到
import TransFTrain as train
import numpy as np

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

dilate_backward_params = [
    {"shape": (2, 5),          "d": 1, "axes": (0,)},
    {"shape": (2, 5),          "d": 2, "axes": (1,)},
    {"shape": (2, 5),          "d": 1, "axes": (0,1)},
    {"shape": (2, 5),          "d": 0, "axes": (0,1)},
    {"shape": (2, 3, 3, 4),     "d": 2, "axes": (0,1)},
    {"shape": (3, 3, 6, 4),     "d": 3, "axes": (0,1)},
    {"shape": (2, 3, 3, 4),     "d": 0, "axes": (1,2)},
    {"shape": (2, 3, 3, 4),     "d": 1, "axes": (1,2)},
    {"shape": (3, 3, 6, 4),     "d": 1, "axes": (1,2)},
    {"shape": (2, 3, 3, 4),     "d": 1, "axes": (2,3)},
    {"shape": (3, 3, 6, 4),     "d": 1, "axes": (2,3)},
    {"shape": (2, 3, 3, 4),     "d": 1, "axes": (0,1,2,3)},
]

class TestDilate(unittest.TestCase):
    def test_dilate_forward_cpu(device):
        device = train.cpu()
        np.random.seed(0)
        device = train.cpu()

        _A = np.random.randint(1, 10, size=(2, 5))
        A = train.Tensor(_A, device=device)
        assert np.linalg.norm(train.dilate(A, dilation=0, axes=(0,)).numpy() - np.array([[6., 1., 4., 4., 8.],
        [4., 6., 3., 5., 8.]])) < 1e-5 

        _A = np.random.randint(1, 10, size=(2, 5))
        A = train.Tensor(_A, device=device)
        assert np.linalg.norm(train.dilate(A, dilation=1, axes=(0,)).numpy() - np.array([[7., 9., 9., 2., 7.],
        [0., 0., 0., 0., 0.],
        [8., 8., 9., 2., 6.],
        [0., 0., 0., 0., 0.]])) < 1e-5

        _A = np.random.randint(1, 10, size=(2, 5))
        A = train.Tensor(_A, device=device)
        assert np.linalg.norm(train.dilate(A, dilation=1, axes=(1,)).numpy() - np.array([[9., 0., 5., 0., 4., 0., 1., 0., 4., 0.],
        [6., 0., 1., 0., 3., 0., 4., 0., 9., 0.]])) < 1e-5

        _A = np.random.randint(1, 10, size=(2, 5))
        A = train.Tensor(_A, device=device)
        assert np.linalg.norm(train.dilate(A, dilation=1, axes=(0,1)).numpy() - np.array([[2., 0., 4., 0., 4., 0., 4., 0., 8., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [1., 0., 2., 0., 1., 0., 5., 0., 8., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])) < 1e-5

        _A = np.random.randint(1, 10, size=(2, 2))
        A = train.Tensor(_A, device=device)
        assert np.linalg.norm(train.dilate(A, dilation=2, axes=(0,1)).numpy() - np.array([[4., 0., 0., 3., 0., 0.],
        [0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.],
        [8., 0., 0., 3., 0., 0.],
        [0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.]])) < 1e-5

        _A = np.random.randint(1, 10, size=(2, 2, 2, 2))
        A = train.Tensor(_A, device=device)
        assert np.linalg.norm(train.dilate(A, dilation=1, axes=(1,2)).numpy() - np.array([[[[1., 1.],
            [0., 0.],
            [5., 6.],
            [0., 0.]],

            [[0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.]],

            [[6., 7.],
            [0., 0.],
            [9., 5.],
            [0., 0.]],

            [[0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.]]],


        [[[2., 5.],
            [0., 0.],
            [9., 2.],
            [0., 0.]],

            [[0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.]],

            [[2., 8.],
            [0., 0.],
            [4., 7.],
            [0., 0.]],

            [[0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.]]]])) < 1e-5

    @unittest.skipIf(not train.cuda().enabled(), "NO GPU")
    def test_dilate_forward_cpu(device):
        device = train.cuda()
        np.random.seed(0)
        device = train.cpu()

        _A = np.random.randint(1, 10, size=(2, 5))
        A = train.Tensor(_A, device=device)
        assert np.linalg.norm(train.dilate(A, dilation=0, axes=(0,)).numpy() - np.array([[6., 1., 4., 4., 8.],
        [4., 6., 3., 5., 8.]])) < 1e-5 

        _A = np.random.randint(1, 10, size=(2, 5))
        A = train.Tensor(_A, device=device)
        assert np.linalg.norm(train.dilate(A, dilation=1, axes=(0,)).numpy() - np.array([[7., 9., 9., 2., 7.],
        [0., 0., 0., 0., 0.],
        [8., 8., 9., 2., 6.],
        [0., 0., 0., 0., 0.]])) < 1e-5

        _A = np.random.randint(1, 10, size=(2, 5))
        A = train.Tensor(_A, device=device)
        assert np.linalg.norm(train.dilate(A, dilation=1, axes=(1,)).numpy() - np.array([[9., 0., 5., 0., 4., 0., 1., 0., 4., 0.],
        [6., 0., 1., 0., 3., 0., 4., 0., 9., 0.]])) < 1e-5

        _A = np.random.randint(1, 10, size=(2, 5))
        A = train.Tensor(_A, device=device)
        assert np.linalg.norm(train.dilate(A, dilation=1, axes=(0,1)).numpy() - np.array([[2., 0., 4., 0., 4., 0., 4., 0., 8., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [1., 0., 2., 0., 1., 0., 5., 0., 8., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])) < 1e-5

        _A = np.random.randint(1, 10, size=(2, 2))
        A = train.Tensor(_A, device=device)
        assert np.linalg.norm(train.dilate(A, dilation=2, axes=(0,1)).numpy() - np.array([[4., 0., 0., 3., 0., 0.],
        [0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.],
        [8., 0., 0., 3., 0., 0.],
        [0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.]])) < 1e-5

        _A = np.random.randint(1, 10, size=(2, 2, 2, 2))
        A = train.Tensor(_A, device=device)
        assert np.linalg.norm(train.dilate(A, dilation=1, axes=(1,2)).numpy() - np.array([[[[1., 1.],
            [0., 0.],
            [5., 6.],
            [0., 0.]],

            [[0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.]],

            [[6., 7.],
            [0., 0.],
            [9., 5.],
            [0., 0.]],

            [[0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.]]],


        [[[2., 5.],
            [0., 0.],
            [9., 2.],
            [0., 0.]],

            [[0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.]],

            [[2., 8.],
            [0., 0.],
            [4., 7.],
            [0., 0.]],

            [[0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.]]]])) < 1e-5
        
    def test_dilate_backward_cpu(self):
        device = train.cpu()
        for params in dilate_backward_params:
            np.random.seed(0)
            shape, d, axes = params['shape'], params['d'], params['axes']
            backward_check(train.dilate, train.Tensor(np.random.randn(*shape), device=device), dilation=d, axes=axes)

    @unittest.skipIf(not train.cuda().enabled(), "NO GPU")
    def test_dilate_backward_cuda(self):
        device = train.cuda()
        for params in dilate_backward_params:
            np.random.seed(0)
            shape, d, axes = params['shape'], params['d'], params['axes']
            backward_check(train.dilate, train.Tensor(np.random.randn(*shape), device=device), dilation=d, axes=axes)


if __name__ == '__main__':
    unittest.main()