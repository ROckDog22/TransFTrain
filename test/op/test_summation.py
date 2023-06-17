import unittest

import sys
sys.path.append('./python')
import TransFTrain as train
 
class TestSummation(unittest.TestCase):
    def setUp(self) -> None:
        self.x = train.Tensor([1,2,3], dtype="int8")

    def test_case1(self):
        a = train.summ

        self.assertEqual(a, b)
    def test_case1(self):
        train.Summation(self.x, 4)

if __name__ == '__main__':
    unittest.main()