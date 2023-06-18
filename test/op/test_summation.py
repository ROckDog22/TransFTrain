import unittest

import sys
sys.path.append('./python')
import TransFTrain as train
 
class TestSummation(unittest.TestCase):
    def setUp(self) -> None:
        self.x = train.Tensor([[1,2,3], [3,2,1]], dtype="int8")

    def test_case1(self):
        y = train.Tensor([12], dtype="int8")
        self.assertEqual(self.x.sum(), y)
    
    def test_case2(self):
        y = train.Tensor([6,6], dtype="int8")
        self.assertEqual(train.summation(self.x, 1), y)
        
if __name__ == '__main__':
    unittest.main()