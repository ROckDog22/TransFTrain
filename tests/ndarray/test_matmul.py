import unittest

import sys
sys.path.append('./python')
# 你需要在.vscode里面添加extra地址 才能找到
import TransFTrain as train
import TransFTrain.backend_ndarray as nd
import numpy as np


matmul_dims = [(16, 16, 16), 
    (8, 8, 8), 
    (1, 2, 3), 
    (3, 4, 5), 
    (5, 4, 3), 
    (64, 64, 64), 
    (72, 72, 72), 
    (72, 73, 74), 
    (74, 73, 72), 
    (128, 128, 128)]
matmul_tiled_shapes = [(1, 1, 1), (2, 2, 3), (1, 2, 1), (3, 3, 3)]
class TestMatMul(unittest.TestCase):
    def test_case1_cpu(self):
        for m, n, p in matmul_dims:
            _A = np.random.randn(m, n)
            _B = np.random.randn(n, p)
            A = nd.array(_A, device=nd.cpu())
            B = nd.array(_B, device=nd.cpu())
            np.testing.assert_allclose((A @ B).numpy(), _A @ _B, rtol=1e-5, atol=1e-5)


    @unittest.skipIf(not nd.cuda().enabled(), "NO GPU")
    def test_case1_cuda(self):
        for m, n, p in matmul_dims:
            _A = np.random.randn(m, n)
            _B = np.random.randn(n, p)
            A = nd.array(_A, device=nd.cuda())
            B = nd.array(_B, device=nd.cuda())
            np.testing.assert_allclose((A @ B).numpy(), _A @ _B, rtol=1e-5, atol=1e-5)

    def test_case2_cpu(self):
        for m, n, p in matmul_tiled_shapes:
            device = nd.cpu()
            assert hasattr(device, "matmul_tiled")
            t = device.__tile_size__
            A = nd.array(np.random.randn(m, n, t, t), device=nd.cpu())
            B = nd.array(np.random.randn(n, p, t, t), device=nd.cpu())
            C = nd.NDArray.make((m, p, t, t), device=nd.cpu())
            device.matmul_tiled(A._handle, B._handle, C._handle, m*t, n*t, p*t)

            lhs = A.numpy().transpose(0, 2, 1, 3).flatten().reshape(m*t, n*t) \
                @ B.numpy().transpose(0, 2, 1, 3).flatten().reshape(n*t, p*t)
            rhs = C.numpy().transpose(0, 2, 1, 3).flatten().reshape(m*t, p*t)

            np.testing.assert_allclose(lhs, rhs, atol=1e-5, rtol=1e-5)


    # @unittest.skipIf(not nd.cuda().enabled(), "NO GPU")
    # def test_case2_cuda(self):
    #     for m, n, p in matmul_tiled_shapes:
    #         device = nd.cpu()
    #         assert hasattr(device, "matmul_tiled")
    #         t = device.__tile_size__
    #         A = nd.array(np.random.randn(m, n, t, t), device=nd.cuda())
    #         B = nd.array(np.random.randn(n, p, t, t), device=nd.cuda())
    #         C = nd.NDArray.make((m, p, t, t), device=nd.cpu())
    #         device.matmul_tiled(A._handle, B._handle, C._handle, m*t, n*t, p*t)

    #         lhs = A.numpy().transpose(0, 2, 1, 3).flatten().reshape(m*t, n*t) \
    #             @ B.numpy().transpose(0, 2, 1, 3).flatten().reshape(n*t, p*t)
    #         rhs = C.numpy().transpose(0, 2, 1, 3).flatten().reshape(m*t, p*t)

    #         np.testing.assert_allclose(lhs, rhs, atol=1e-5, rtol=1e-5)


if "__main__" == __name__:
    unittest.main()
