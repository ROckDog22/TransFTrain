import unittest

import sys
sys.path.append('./python')
# 你需要在.vscode里面添加extra地址 才能找到
import TransFTrain as train

class TestEwiseAdd(unittest.TestCase):
    def test_case1(self):
        x = train.Tensor([1,2,3], dtype="int8")
        y = train.Tensor([4,5,6], dtype="int8")
        z = train.Tensor([5,7,9], dtype="int8")
        self.assertEqual(train.add(x, y).data, z.data)

    def test_case2(self):
        x = train.Tensor([[1,2,3],[1,2,3]], dtype="int32")
        y = train.Tensor([[4,5,6], [4,5,6]], dtype="int32")
        z = train.Tensor([[5,7,9], [5,7,9]], dtype="int32")
        self.assertEqual(train.add(x, y), z)
    
    def test_case3(self):
        x = train.Tensor([[1,2,3],[1,2,3]], dtype="int8")
        y = train.Tensor([[4,5,6], [4,5,6]], dtype="int8")
        z = train.Tensor([[5,7,9], [5,7,9]], dtype="int8")
        self.assertEqual(x + y, z)

    def test_case4(self):
        x = train.Tensor([[1,2,3],[1,2,3]], dtype="int8")
        y = train.Tensor([[4,5,6], [4,5,6]], dtype="int8")
        z = train.Tensor([[5,7,9], [5,7,9]], dtype="int8")
        self.assertEqual(train.add(x, y), z)
        
    def test_case5(self):
        x = train.Tensor([[1,2,3],[1,2,3]], dtype="int8")
        y = train.Tensor([[4,5,6], [4,5,6]], dtype="int8")
        z = train.Tensor([[5,7,9], [5,7,9]], dtype="int8")
        self.assertEqual(train.add(x, y), z)

if __name__ == '__main__':
    unittest.main()