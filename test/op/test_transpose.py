import unittest

import sys
sys.path.append('./python')
# 你需要在.vscode里面添加extra地址 才能找到
import TransFTrain as train

class TestTranspose(unittest.TestCase):
    def test_case1(self):
        x = train.Tensor([[1,2,3],[1,2,3]] , dtype="int8")
        y = train.Tensor([[1,1],[2,2],[3,3]], dtype="int8")
        z = x.transpose((0,1))
        self.assertEqual(z,y)
        self.assertEqual(z.shape, (3,2))

    def test_case2(self):
        x = train.Tensor([[[1,2,3],[1,2,3]], [[1,54,3],[1,1,3]]] , dtype="int8")
        y = train.Tensor([[[1,1],[1,1]],[[2,54],[2,1]],[[3,3],[3,3]]], dtype="int8")
        z = x.transpose((0,2))
        self.assertEqual(z,y)
        self.assertEqual(z.shape, (3,2,2))

    def test_case3(self):
        x = train.Tensor([[1,2,3],[1,2,3]] , dtype="int8")
        y = train.Tensor([[1,1],[2,2],[3,3]], dtype="int8")
        z = train.transpose(x, (0,1))
        self.assertEqual(z,y)
        self.assertEqual(z.shape, (3,2))

    def test_case4(self):
        x = train.Tensor([[[1,2,3],[1,2,3]], [[1,54,3],[1,1,3]]] , dtype="int8")
        y = train.Tensor([[[1,1],[1,1]],[[2,54],[2,1]],[[3,3],[3,3]]], dtype="int8")
        z = train.transpose(x, (0,2))
        self.assertEqual(z,y)
        self.assertEqual(z.shape, (3,2,2))

if __name__ == '__main__':
    unittest.main()
    