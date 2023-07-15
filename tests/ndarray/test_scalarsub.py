import unittest

import sys
sys.path.append('./python')
# 你需要在.vscode里面添加extra地址 才能找到
import TransFTrain as train
import TransFTrain.backend_ndarray as nd
import numpy as np

GENERAL_SHAPES = [(1, 1, 1), (4, 5, 6)]

class TestScalarSub(unittest.TestCase):
    def test_scalar_fn_cpu(self):
        device = train.cpu()
        fn = lambda a, b: a - b
        for shape in GENERAL_SHAPES:
            _A = np.random.randn(*shape).astype(np.float32)
            _B = np.random.randn(1).astype(np.float32).item()
            A = train.Tensor(nd.array(_A), device=device)
            np.testing.assert_allclose(fn(_A, _B), fn(A, _B).numpy(), atol=1e-5, rtol=1e-5)

    @unittest.skipIf(not train.cuda().enabled(), "NO GPU")
    def test_scalar_fn_cuda(self):
        device = train.cuda()
        fn = lambda a, b: a - b
        for shape in GENERAL_SHAPES:
            _A = np.random.randn(*shape).astype(np.float32)
            _B = np.random.randn(1).astype(np.float32).item()
            A = train.Tensor(nd.array(_A), device=device)
            np.testing.assert_allclose(fn(_A, _B), fn(A, _B).numpy(), atol=1e-5, rtol=1e-5)

if "__main__" == __name__:
    unittest.main()
