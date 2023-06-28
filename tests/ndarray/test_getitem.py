import unittest

import sys
sys.path.append('./python')
# 你需要在.vscode里面添加extra地址 才能找到
import TransFTrain as train
import TransFTrain.backend_ndarray as nd
import numpy as np
class TestPermute(unittest.TestCase):
    def setUp(self):
        self.x = np.array([[[[1,2,3],[1,2,3]]]])

    def test_case1(self):
        x = nd.NDArray(self.x)
        z = x.permute((0,3,1,2))
        self.assertEqual(x.strides, (6, 6, 3, 1))
        self.assertEqual(x.shape, (1, 1, 2, 3))
        self.assertEqual(z.strides, (6, 1, 6, 3))
        self.assertEqual(z.shape, (1, 3, 1, 2))

if __name__ == '__main__':
    unittest.main()