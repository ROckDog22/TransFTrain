import unittest

import sys
sys.path.append('./python')
# 你需要在.vscode里面添加extra地址 才能找到
import TransFTrain as train

class TestMatMul(unittest.TestCase):
    def test_case1(self):
        x = train.Tensor([[1,2], [2, 3]], dtype="int8")
        y = train.Tensor([[[1,2], [2, 3]], [[4,2], [3, 3]]], dtype="int8")
        z = train.Tensor([[[5,8],[8,13]], [[10,8],[17,13]]])
        self.assertEqual(x.matmul(y), z)
    
    def test_case2(self):
        x = train.Tensor([[1,2], [2, 3]], dtype="int8")
        y = train.Tensor([[[1,2], [2, 3]], [[4,2], [3, 3]]], dtype="int8")
        z = train.Tensor([[[5,8],[8,13]], [[10,8],[17,13]]])
        self.assertEqual(train.matmul(x,y), z)

if __name__ == '__main__':
    unittest.main()