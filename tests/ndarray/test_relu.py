import unittest

import sys
sys.path.append('./python')
# 你需要在.vscode里面添加extra地址 才能找到
import TransFTrain as train
import TransFTrain.backend_ndarray as nd
import numpy as np

GENERAL_SHAPES = [(1, 1, 1), (4, 5, 6)]
class TestRule(unittest.TestCase):
    def test_relu_cpu(self):
        device = train.cpu()
        for shape in GENERAL_SHAPES:
            _A = np.random.randn(*shape).astype(np.float32)
            A = train.Tensor(nd.array(_A), device=device)
            np.testing.assert_allclose(np.maximum(_A, 0), train.relu(A).numpy(), atol=1e-5, rtol=1e-5)
    
    @unittest.skipIf(not nd.cuda().enabled(), "NO GPU")
    def test_relu_cuda(self):
        device = train.cuda()
        for shape in GENERAL_SHAPES:
            _A = np.random.randn(*shape).astype(np.float32)
            A = train.Tensor(nd.array(_A), device=device)
            np.testing.assert_allclose(np.maximum(_A, 0), train.relu(A).numpy(), atol=1e-5, rtol=1e-5)

if "__main__" == __name__:
    unittest.main()
