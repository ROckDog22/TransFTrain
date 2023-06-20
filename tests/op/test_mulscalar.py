import unittest

import sys
sys.path.append('./python')
# 你需要在.vscode里面添加extra地址 才能找到
import TransFTrain as train

class TestMulScalar(unittest.TestCase):
    def test_case1(self):
        x = train.Tensor([1,1,1], dtype="int8")
        y = 5
        z = train.Tensor([5,5,5], dtype="int8")
        self.assertEqual(x*y, z)

    def test_case2(self):
        x = train.Tensor([1,1,1], dtype="int8")
        y = 5
        z = train.Tensor([5,5,5], dtype="int8")
        self.assertEqual(train.mul_scalar(x,y), z)

if __name__ == '__main__':
    unittest.main()