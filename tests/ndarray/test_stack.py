import unittest

import sys
sys.path.append('./python')
# 你需要在.vscode里面添加extra地址 才能找到
import TransFTrain as train
import TransFTrain.backend_ndarray as nd
import numpy as np
import torch

STACK_PARAMETERS = [((5, 5), 0, 1),
    ((5, 5), 0, 2),
    ((1,5,7), 2, 5)]

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
    def test_stack_cpu(self):
        device = train.cpu()
        for shape, axis, l in STACK_PARAMETERS:
            _A = [np.random.randn(*shape).astype(np.float32) for i in range(l)]
            A = [train.Tensor(nd.array(_A[i]), device=device) for i in range(l)]
            A_t = [torch.Tensor(_A[i]) for i in range(l)]
            out = train.stack(A, axis=axis)
            out_t = torch.stack(A_t, dim=axis)
            np.testing.assert_allclose(out_t.numpy(), out.numpy(), atol=1e-5, rtol=1e-5)

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
                
if __name__=="__main__":
    unittest.main()