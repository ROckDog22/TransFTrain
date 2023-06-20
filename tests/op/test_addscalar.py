import unittest

import sys
sys.path.append('./python')
# 你需要在.vscode里面添加extra地址 才能找到
import TransFTrain as train

class TestAddScalar(unittest.TestCase):
    def test_case1(self):
        x = train.Tensor([1,2,3], dtype="int8")
        y = 5
        z = train.Tensor([6,7,8], dtype="int8")
        self.assertEqual(x+y, z)

    def test_case2(self):
        x = train.Tensor([1,2,3], dtype="int8")
        y = 5
        z = train.Tensor([6,7,8], dtype="int8")
        self.assertEqual(train.add_scalar(x,y), z)

if __name__ == '__main__':
    unittest.main()