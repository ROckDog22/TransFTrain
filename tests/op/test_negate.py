import unittest

import sys
sys.path.append('./python')
# 你需要在.vscode里面添加extra地址 才能找到
import TransFTrain as train

class Testnegate(unittest.TestCase):
    def test_case1(self):
        x = train.Tensor([1,1,1], dtype="int8")
        y = train.negate(x)
        z = train.Tensor([-1,-1,-1], dtype="int8")
        self.assertEqual(y, z)

    def test_case2(self):
        x = train.Tensor([1,1,1], dtype="int8")
        z = train.Tensor([-1,-1,-1], dtype="int8")
        self.assertEqual(-x, z)

if __name__ == '__main__':
    unittest.main()