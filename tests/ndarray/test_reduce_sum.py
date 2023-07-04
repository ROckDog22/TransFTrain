import unittest

import sys
sys.path.append('./python')
# 你需要在.vscode里面添加extra地址 才能找到
import TransFTrain as train
import TransFTrain.backend_ndarray as nd
import numpy as np

reduce_params = [
    {"dims": (10,), "axis": 0},
    {"dims": (4, 5, 6), "axis": 0},
    {"dims": (4, 5, 6), "axis": 1},
    {"dims": (4, 5, 6), "axis": 2}
]

class TestReduceSum(unittest.TestCase):
    def test_case1_cpu(self):
        for params in reduce_params:
            dims, axis = params['dims'], params['axis']
            _A = np.random.randn(*dims)
            A = nd.array(_A, device=nd.cpu())   
            np.testing.assert_allclose(_A.sum(axis=axis, keepdims=True), A.sum(axis=axis).numpy(), atol=1e-5, rtol=1e-5)


    @unittest.skipIf(not nd.cuda().enabled(), "NO GPU")
    def test_case1_cuda(self):
        for params in reduce_params:
            dims, axis = params['dims'], params['axis']
            _A = np.random.randn(*dims)
            A = nd.array(_A, device=nd.cuda())   
            np.testing.assert_allclose(_A.sum(axis=axis, keepdims=True), A.sum(axis=axis).numpy(), atol=1e-5, rtol=1e-5)

if "__main__" == __name__:
    unittest.main()
