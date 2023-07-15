import unittest

import sys
sys.path.append('./python')
# 你需要在.vscode里面添加extra地址 才能找到
import TransFTrain as train
import TransFTrain.backend_ndarray as nd
import numpy as np


GENERAL_SHAPES = [(1, 1, 1), (4, 5, 6)]
class TestEwiseExp(unittest.TestCase):
    def test_case1_cpu(self):
        A = np.random.randn(5, 5)
        B = nd.array(A, device=nd.cpu())
        np.testing.assert_allclose(np.exp(A), (B.exp()).numpy(), atol=1e-5, rtol=1e-5)


    @unittest.skipIf(not train.cuda().enabled(), "NO GPU")
    def test_case1_cuda(self):
        A = np.random.randn(5, 5)
        B = nd.array(A, device=train.cuda())
        np.testing.assert_allclose(np.exp(A), (B.exp()).numpy(), atol=1e-5, rtol=1e-5)


    def test_exp_cpu(self):
        device = train.cpu()
        for shape in GENERAL_SHAPES:
            _A = np.random.randn(*shape).astype(np.float32)
            A = train.Tensor(nd.array(_A), device=device)
            np.testing.assert_allclose(np.exp(_A), train.exp(A).numpy(), atol=1e-5, rtol=1e-5)


    @unittest.skipIf(not train.cuda().enabled(), "NO GPU")
    def test_exp_cuda(self):
        device = train.cuda()
        for shape in GENERAL_SHAPES:
            _A = np.random.randn(*shape).astype(np.float32)
            A = train.Tensor(nd.array(_A), device=device)
            np.testing.assert_allclose(np.exp(_A), train.exp(A).numpy(), atol=1e-5, rtol=1e-5)

if "__main__" == __name__:
    unittest.main()
