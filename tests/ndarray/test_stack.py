import unittest

import sys
sys.path.append('./python')
import TransFTrain as train
import TransFTrain.backend_ndarray as nd
import numpy as np
import torch

STACK_PARAMETERS = [((5, 5), 0, 1),
    ((5, 5), 0, 2),
    ((1,5,7), 2, 5)]

stack_params = [
    {"shape": (10,3),    "n": 4, "axis": 0},
    {"shape": (4, 5, 6), "n": 5, "axis": 0},
    {"shape": (4, 5, 6), "n": 3, "axis": 1},
    {"shape": (4, 5, 6), "n": 2, "axis": 2}
]

stack_back_params = [
    ( (3, 4), 3, 0),
    ( (3, 4), 3, 1),
    ( (3, 4), 3, 2),
    ( (3, 4), 5, 2),
    ( (3, 4), 1, 2),
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

def backward_check_post(f, *args, **kwargs):
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

class TestStack(unittest.TestCase):
    def test_stack_cpu(self):
        device = train.cpu()
        for shape, axis, l in STACK_PARAMETERS:
            _A = [np.random.randn(*shape).astype(np.float32) for i in range(l)]
            A = [train.Tensor(nd.array(_A[i]), device=device) for i in range(l)]
            A_t = [torch.Tensor(_A[i]) for i in range(l)]
            out = train.stack(A, axis=axis)
            out_t = torch.stack(A_t, dim=axis)
            np.testing.assert_allclose(out_t.numpy(), out.numpy(), atol=1e-5, rtol=1e-5)

    @unittest.skipIf(not train.cuda().enabled(), "NO GPU")
    def test_stack_cuda(self):
        device = train.cuda()
        for shape, axis, l in STACK_PARAMETERS:
            _A = [np.random.randn(*shape).astype(np.float32) for i in range(l)]
            A = [train.Tensor(nd.array(_A[i]), device=device) for i in range(l)]
            A_t = [torch.Tensor(_A[i]) for i in range(l)]
            out = train.stack(A, axis=axis)
            out_t = torch.stack(A_t, dim=axis)
            np.testing.assert_allclose(out_t.numpy(), out.numpy(), atol=1e-5, rtol=1e-5)

    def test_stack_backward_cpu(self):
        device = train.cpu()
        for shape, axis, l in STACK_PARAMETERS:
            _A = [np.random.randn(*shape).astype(np.float32) for i in range(l)]
            A = [train.Tensor(nd.array(_A[i]), device=device) for i in range(l)]
            A_t = [torch.Tensor(_A[i]) for i in range(l)]
            for i in range(l):
                A_t[i].requires_grad = True
            train.stack(A, axis=axis).sum().backward()
            torch.stack(A_t, dim=axis).sum().backward()
            for i in range(l):
                np.testing.assert_allclose(A_t[i].grad.numpy(), A[i].grad.numpy(), atol=1e-5, rtol=1e-5)

    @unittest.skipIf(not train.cuda().enabled(), "NO GPU")
    def test_stack_backward_cuda(self):
        device = train.cuda()
        for shape, axis, l in STACK_PARAMETERS:
            _A = [np.random.randn(*shape).astype(np.float32) for i in range(l)]
            A = [train.Tensor(nd.array(_A[i]), device=device) for i in range(l)]
            A_t = [torch.Tensor(_A[i]) for i in range(l)]
            for i in range(l):
                A_t[i].requires_grad = True
            train.stack(A, axis=axis).sum().backward()
            torch.stack(A_t, dim=axis).sum().backward()
            for i in range(l):
                np.testing.assert_allclose(A_t[i].grad.numpy(), A[i].grad.numpy(), atol=1e-5, rtol=1e-5)
    
    def test_stack_backward_cpu_post(self):
        np.random.seed(0)
        device = train.cpu()
        for params in stack_params:
            shape, n, axis = params["shape"],params["n"],params["axis"]
            get_tensor = lambda shape: train.Tensor(np.random.randn(*shape)*5, device=device)
            backward_check_post(train.stack, [get_tensor(shape) for _ in range(n)], axis=axis)


    @unittest.skipIf(not train.cuda().enabled(), "NO GPU")
    def test_stack_backward_cuda_post(self):
        np.random.seed(0)
        device = train.cuda()
        for params in stack_params:
            shape, n, axis = params["shape"],params["n"],params["axis"]
            get_tensor = lambda shape: train.Tensor(np.random.randn(*shape)*5, device=device)
            backward_check_post(train.stack, [get_tensor(shape) for _ in range(n)], axis=axis)

    def test_stack_forward_cpu_post1(self):
        np.random.seed(0)
        device = train.cpu()
        for params in stack_params:
            np.random.seed(0)
            shape, n, axis = params['shape'], params['n'], params['axis']
            to_stack_ndl = []
            to_stack_npy = []
            for i in range(n):
                _A = np.random.randn(*shape)
                to_stack_ndl += [train.Tensor(_A, device=device)]
                to_stack_npy += [_A]

            lhs = np.stack(to_stack_npy, axis=axis)
            rhs = train.stack(to_stack_ndl, axis=axis)


    @unittest.skipIf(not train.cuda().enabled(), "NO GPU")
    def test_stack_backward_cuda_post1(self):
        np.random.seed(0)
        device = train.cuda()
        for params in stack_params:
            np.random.seed(0)
            shape, n, axis = params['shape'], params['n'], params['axis']
            to_stack_ndl = []
            to_stack_npy = []
            for i in range(n):
                _A = np.random.randn(*shape)
                to_stack_ndl += [train.Tensor(_A, device=device)]
                to_stack_npy += [_A]

            lhs = np.stack(to_stack_npy, axis=axis)
            rhs = train.stack(to_stack_ndl, axis=axis)


    def test_stack_vs_pytorch(self):
        np.random.seed(0)
        import torch
        A = np.random.randn(5, 5)
        B = np.random.randn(5, 5)
        C = np.random.randn(5, 5)
        D = np.random.randn(15, 5)

        Atrain = train.Tensor(A, requires_grad=True)
        Btrain = train.Tensor(B, requires_grad=True)
        Ctrain = train.Tensor(C, requires_grad=True)
        Dtrain = train.Tensor(D, requires_grad=True)

        Atch = torch.tensor(A, requires_grad=True)
        Btch = torch.tensor(B, requires_grad=True)
        Ctch = torch.tensor(C, requires_grad=True)
        Dtch = torch.tensor(D, requires_grad=True)

        Xtrain = train.stack([Atrain, Ctrain @ Btrain, Ctrain], axis=1)
        Xtch = torch.stack([Atch, Ctch @ Btch, Ctch], dim=1)

        assert Xtrain.shape == Xtch.shape
        assert np.linalg.norm(Xtrain.numpy() - Xtch.detach().numpy()) < 1e-3

        Ytrain = (Dtrain @ Xtrain.reshape((5, 15)) @ Dtrain).sum()
        Ytch = (Dtch @ Xtch.reshape(5, 15) @ Dtch).sum()

        assert np.linalg.norm(Ytrain.numpy() - Ytch.detach().numpy()) < 1e-3

        Ytrain.backward()
        Ytch.backward()

        assert np.linalg.norm(Atrain.grad.cached_data.numpy() - Atch.grad.detach().numpy()) < 1e-3
        assert np.linalg.norm(Btrain.grad.cached_data.numpy() - Btch.grad.detach().numpy()) < 1e-3
        assert np.linalg.norm(Ctrain.grad.cached_data.numpy() - Ctch.grad.detach().numpy()) < 1e-3

if __name__=="__main__":
    unittest.main()