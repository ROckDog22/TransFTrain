import unittest

import sys
sys.path.append('./python')
# 你需要在.vscode里面添加extra地址 才能找到
import TransFTrain as train

class TestRulu(unittest.TestCase):
    def test_case1(self):
        x = train.Tensor([2,3,4], dtype="int8")
        z = train.Tensor([2,3,4], dtype="int8")
        self.assertEqual(train.relu(x), z)

    def test_case2(self):
        x = train.Tensor([2,3,-4], dtype="int8")
        z = train.Tensor([2,3,0], dtype="int8")
        self.assertEqual(train.relu(x), z)

if __name__ == '__main__':
    unittest.main()