import unittest

import sys
sys.path.append('./python')
# 你需要在.vscode里面添加extra地址 才能找到
import TransFTrain as train

class TestBraodcast_to(unittest.TestCase):
    def test_case1(self):
        x = train.Tensor([1,2,3], dtype="int8")
        z = train.Tensor([[1,2,3],[1,2,3]], dtype="int8")
        self.assertEqual(x.broadcast_to((2,3)), z)
    
    def test_case2(self):
        x = train.Tensor([1,2,3], dtype="int8")
        z = train.Tensor([[1,2,3],[1,2,3]], dtype="int8")
        self.assertEqual(train.broadcast_to(x, (2,3)), z)

if __name__ == '__main__':
    unittest.main()