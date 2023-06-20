import unittest

import sys
sys.path.append('./python')
# 你需要在.vscode里面添加extra地址 才能找到
import TransFTrain as train

class TestReshape(unittest.TestCase):
    def test_case1(self):
        x = train.Tensor([[1,2,3],[1,2,3]] , dtype="int8")
        y = train.Tensor([[1,2],[3,1],[2,3]], dtype="int8")
        z = x.reshape((3,2))
        self.assertEqual(z,y)

    def test_case2(self):
        x = train.Tensor([[1,2,3],[1,2,3]] , dtype="int8")
        y = train.Tensor([[1,2],[3,1],[2,3]], dtype="int8")
        z = train.reshape(y, (2,3))
        self.assertEqual(x,z)

if __name__ == '__main__':
    unittest.main()
    