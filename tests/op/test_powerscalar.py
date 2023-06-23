import unittest
import numpy as np

import sys
sys.path.append('./python')
# 你需要在.vscode里面添加extra地址 才能找到
import TransFTrain as train

class TestPowerScalar(unittest.TestCase):
    def test_case1(self):
        x = train.Tensor([2,3,4], dtype="int8")
        z = train.Tensor([pow(2, 3), pow(3, 3), pow(4, 3)], dtype="int8")
        self.assertEqual(train.power_scalar(x, 3), z)

    def test_case2(self):
        x = train.Tensor([1,1,1], dtype="int8")
        y = 5
        z = train.Tensor([5,5,5], dtype="int8")
        self.assertEqual(train.mul_scalar(x,y), z)
    

    def get_tensor(self, *shape, entropy=1):
        np.random.seed(np.prod(shape) * len(shape) * entropy)
        return train.Tensor(np.random.randint(0, 100, size=shape) / 20, dtype="float32")

    def get_int_tensor(self, *shape, low=0, high=10, entropy=1):
        np.random.seed(np.prod(shape) * len(shape) * entropy)
        return train.Tensor(np.random.randint(low, high, size=shape))

    def power_scalar_forward(self, shape, power=2):
        x = self.get_tensor(*shape)
        return (x**power).cached_data

    def power_scalar_backward(self, shape, power=2):
        x = self.get_tensor(*shape)
        y = (x**power).sum()
        y.backward()
        return x.grad.cached_data
    
    def test_op_power_scalar_forward_1(self):
        np.testing.assert_allclose(self.power_scalar_forward((2,2), power=2),
            np.array([[11.222499, 17.639997],
            [ 0.0625 , 20.25 ]], dtype=np.float32), rtol=1e-5, atol=1e-5)

    def test_op_power_scalar_forward_2(self):
        np.testing.assert_allclose(self.power_scalar_forward((2,2), power=-1.5),
            np.array([[0.16309206, 0.11617859],
            [8. , 0.10475656]], dtype=np.float32), rtol=1e-5, atol=1e-5)

    def test_op_power_scalar_backward_1(self):
        np.testing.assert_allclose(self.power_scalar_backward((2,2), power=2),
            np.array([[6.7, 8.4],
            [0.5, 9. ]], dtype=np.float32), rtol=1e-5, atol=1e-5)

if __name__ == '__main__':
    unittest.main()