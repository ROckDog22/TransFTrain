import unittest

import sys
sys.path.append('./python')
# 你需要在.vscode里面添加extra地址 才能找到
import TransFTrain as train

class TestExp(unittest.TestCase):
    def test_case1(self):
        x = train.Tensor([1,1,1], dtype="int8")
        y = train.Tensor([0,0,0], dtype="int8")
        self.assertEqual(train.exp(y), x)

    def test_case2(self):
        x = train.Tensor([1,1,1], dtype="int8")
        z = train.Tensor([0,0,0], dtype="int8")
        self.assertEqual(z.exp(), x)

if __name__ == '__main__':
    unittest.main()