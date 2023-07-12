import unittest

import sys
sys.path.append('./python')
# 你需要在.vscode里面添加extra地址 才能找到
import TransFTrain as train

class TestDivScalar(unittest.TestCase):
    def test_case1(self):
        x = train.Tensor([1,2,3], dtype="int8")
        y = 3
        z = train.Tensor([3,6,9], dtype="int8")
        self.assertEqual(z/3, x)

    def test_case2(self):
        x = train.Tensor([1,2,3], dtype="int8")
        y = 3
        z = train.Tensor([3,6,9], dtype="int8")
        self.assertEqual(train.divide_scalar(z,y), x)

if __name__ == '__main__':
    unittest.main()