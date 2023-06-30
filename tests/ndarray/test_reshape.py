import unittest

import sys
sys.path.append('./python')
# 你需要在.vscode里面添加extra地址 才能找到
import TransFTrain as train
import TransFTrain.backend_ndarray as nd
import numpy as np
class TestReshape(unittest.TestCase):
    def test_case1(self):
        shape = (4, 3)
        _A = np.random.randint(low=0, high=10, size=shape)
        A = nd.array(_A, device=nd.cpu())
        lhs = A.reshape((2, 2, 3)).compact()
        assert lhs.is_compact(), "array is not compact"
        rhs = _A.reshape(2, 2, 3)
        np.testing.assert_allclose(lhs.numpy(), rhs, atol=1e-5)

    def test_case2(self):
        shape = (16, 16)
        _A = np.random.randint(low=0, high=10, size=shape)
        A = nd.array(_A, device=nd.cpu())
        lhs = A.reshape((2, 4, 2, 2, 2, 2, 2)).compact()
        assert lhs.is_compact(), "array is not compact"
        rhs = _A.reshape(2, 4, 2, 2, 2, 2, 2)
        np.testing.assert_allclose(lhs.numpy(), rhs, atol=1e-5)

    def test_case3(self):
        shape = (2, 4, 2, 2, 2, 2, 2)
        _A = np.random.randint(low=0, high=10, size=shape)
        A = nd.array(_A, device=nd.cpu())
        lhs = A.reshape((16, 16)).compact()
        assert lhs.is_compact(), "array is not compact"
        rhs = _A.reshape(16, 16)
        np.testing.assert_allclose(lhs.numpy(), rhs, atol=1e-5)

    def test_case4(self):
        shape = (4, 3)
        _A = np.random.randint(low=0, high=10, size=shape)
        A = nd.array(_A, device=nd.cuda())
        lhs = A.reshape((2, 2, 3)).compact()
        assert lhs.is_compact(), "array is not compact"
        rhs = _A.reshape(2, 2, 3)
        np.testing.assert_allclose(lhs.numpy(), rhs, atol=1e-5)

    def test_case5(self):
        shape = (16, 16)
        _A = np.random.randint(low=0, high=10, size=shape)
        A = nd.array(_A, device=nd.cuda())
        lhs = A.reshape((2, 4, 2, 2, 2, 2, 2)).compact()
        assert lhs.is_compact(), "array is not compact"
        rhs = _A.reshape(2, 4, 2, 2, 2, 2, 2)
        np.testing.assert_allclose(lhs.numpy(), rhs, atol=1e-5)

    def test_case6(self):
        shape = (2, 4, 2, 2, 2, 2, 2)
        _A = np.random.randint(low=0, high=10, size=shape)
        A = nd.array(_A, device=nd.cuda())
        lhs = A.reshape((16, 16)).compact()
        assert lhs.is_compact(), "array is not compact"
        rhs = _A.reshape(16, 16)
        np.testing.assert_allclose(lhs.numpy(), rhs, atol=1e-5)

if __name__ == '__main__':
    unittest.main()