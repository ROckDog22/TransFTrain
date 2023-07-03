import unittest

import sys
sys.path.append('./python')
# 你需要在.vscode里面添加extra地址 才能找到
import TransFTrain as train
import TransFTrain.backend_ndarray as nd
import numpy as np
class TestGetitem(unittest.TestCase):
    def test_case1(self):
        shape = (8, 8)
        _A = np.random.randint(low=0, high=10, size=shape)
        A = nd.array(_A, device=nd.cpu())
        lhs = A[4:, 4:].compact()
        assert lhs.is_compact(), "array is not compact"
        rhs = _A[4:, 4:]
        np.testing.assert_allclose(lhs.numpy(), rhs, atol=1e-5)

    def test_case2(self):
        shape = (8, 8, 2, 2, 2, 2)
        _A = np.random.randint(low=0, high=10, size=shape)
        A = nd.array(_A, device=nd.cpu())
        lhs = A[1:3, 5:8, 1:2, 0:1, 0:1, 1:2].compact()
        assert lhs.is_compact(), "array is not compact"
        rhs = _A[1:3, 5:8, 1:2, 0:1, 0:1, 1:2]
        np.testing.assert_allclose(lhs.numpy(), rhs, atol=1e-5)

    def test_case3(self):
        shape = (7, 8)
        _A = np.random.randint(low=0, high=10, size=shape)
        A = nd.array(_A, device=nd.cpu())
        lhs = A.permute((1,0))[3:7,2:5].compact()
        assert lhs.is_compact(), "array is not compact"
        rhs = _A.transpose()[3:7,2:5]
        np.testing.assert_allclose(lhs.numpy(), rhs, atol=1e-5)

    @unittest.skipIf(not nd.cuda().enabled(), "NO GPU")
    def test_case4(self):
        shape = (8, 8)
        _A = np.random.randint(low=0, high=10, size=shape)
        A = nd.array(_A, device=nd.cuda())
        lhs = A[4:, 4:].compact()
        assert lhs.is_compact(), "array is not compact"
        rhs = _A[4:, 4:]
        np.testing.assert_allclose(lhs.numpy(), rhs, atol=1e-5)

    @unittest.skipIf(not nd.cuda().enabled(), "NO GPU")
    def test_case5(self):
        shape = (8, 8, 2, 2, 2, 2)
        _A = np.random.randint(low=0, high=10, size=shape)
        A = nd.array(_A, device=nd.cuda())
        lhs = A[1:3, 5:8, 1:2, 0:1, 0:1, 1:2].compact()
        assert lhs.is_compact(), "array is not compact"
        rhs = _A[1:3, 5:8, 1:2, 0:1, 0:1, 1:2]
        np.testing.assert_allclose(lhs.numpy(), rhs, atol=1e-5)
        
    @unittest.skipIf(not nd.cuda().enabled(), "NO GPU")
    def test_case6(self):
        shape = (7, 8)
        _A = np.random.randint(low=0, high=10, size=shape)
        A = nd.array(_A, device=nd.cuda())
        lhs = A.permute((1,0))[3:7,2:5].compact()
        assert lhs.is_compact(), "array is not compact"
        rhs = _A.transpose()[3:7,2:5]
        np.testing.assert_allclose(lhs.numpy(), rhs, atol=1e-5)

    

if __name__ == '__main__':
    unittest.main()