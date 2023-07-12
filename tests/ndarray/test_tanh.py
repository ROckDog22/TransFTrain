import unittest

import sys
sys.path.append('./python')
# 你需要在.vscode里面添加extra地址 才能找到
import TransFTrain as train
import TransFTrain.backend_ndarray as nd
import numpy as np

GENERAL_SHAPES = [(1, 1, 1), (4, 5, 6)]

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


class Testtanh(unittest.TestCase):
    # def test_case1_cpu(self):
    #     A = np.random.randn(5, 5)
    #     B = nd.array(A, device=nd.cpu())
    #     np.testing.assert_allclose(np.tanh(A), (B.tanh()).numpy(), atol=1e-5, rtol=1e-5)


    # @unittest.skipIf(not nd.cuda().enabled(), "NO GPU")
    # def test_case1_cuda(self):
    #     A = np.random.randn(5, 5)
    #     B = nd.array(A, device=nd.cuda())
    #     np.testing.assert_allclose(np.tanh(A), (B.tanh()).numpy(), atol=1e-5, rtol=1e-5)

        
    # def test_tanh_cpu(self):
    #     device = train.cpu()
    #     for shape in GENERAL_SHAPES:
    #         _A = np.random.randn(*shape).astype(np.float32)
    #         A = train.Tensor(nd.array(_A), device=device)
    #         np.testing.assert_allclose(np.tanh(_A), train.tanh(A).numpy(), atol=1e-5, rtol=1e-5)
    
    # @unittest.skipIf(not nd.cuda().enabled(), "NO GPU")
    # def test_tanh_cuda(self):
    #     device = train.cuda()
    #     for shape in GENERAL_SHAPES:
    #         _A = np.random.randn(*shape).astype(np.float32)
    #         A = train.Tensor(nd.array(_A), device=device)
    #         np.testing.assert_allclose(np.tanh(_A), train.tanh(A).numpy(), atol=1e-5, rtol=1e-5)

    def test_tanh_backward_cpu(self):
        device = train.cpu()
        for shape in GENERAL_SHAPES:
            _A = np.random.randn(*shape).astype(np.float32)
            A = train.Tensor(nd.array(_A), device=device)
            backward_check(train.tanh, A)

    # @unittest.skipIf(not nd.cuda().enabled(), "NO GPU")
    # def test_tanh_backward_cuda(self):
    #     device = train.cuda()
    #     for shape in GENERAL_SHAPES:
    #         _A = np.random.randn(*shape).astype(np.float32)
    #         A = train.Tensor(nd.array(_A), device=device)
    #         backward_check(train.tanh, A)


if "__main__" == __name__:
    unittest.main()
