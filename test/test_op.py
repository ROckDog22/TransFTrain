import unittest

import sys
sys.path.append('./python')
import TransFTrain as train
 
x = train.Tensor([1,2,3], dtype="int8")
y = train.Tensor(x)

def add_numbers(x, y):
    return x + y

class TestSummation(unittest.TestCase):
    def test_addition(self):
        self.assertEqual(add_numbers(2, 3), 5)
        self.assertEqual(add_numbers(-1, 1), 0)
        self.assertEqual(add_numbers(0, 0), 0)

if __name__ == '__main__':
    unittest.main()