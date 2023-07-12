import unittest

import sys
sys.path.append('./python')
# 你需要在.vscode里面添加extra地址 才能找到
import TransFTrain as train

class TestEwiseDiv(unittest.TestCase):
    def test_case1(self):
        x = train.Tensor([1,1,2], dtype="int8")
        y = train.Tensor([4,5,6], dtype="int8")
        z = train.Tensor([4,5,3], dtype="int8")
        self.assertEqual(y / x, z)


    def test_case2(self):
        x = train.Tensor([1,1,2], dtype="int8")
        y = train.Tensor([4,5,6], dtype="int8")
        z = train.Tensor([4,5,3], dtype="int8")
        self.assertEqual(train.divide(y, x), z)
    
if __name__ == '__main__':
    unittest.main()