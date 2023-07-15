import unittest

import sys
sys.path.append('./python')
# 你需要在.vscode里面添加extra地址 才能找到
import TransFTrain as train
import TransFTrain.backend_ndarray as nd
import numpy as np

def compare_strides(a_np, a_nd):
    size = a_np.itemsize
    assert tuple([x // size for x in a_np.strides]) == a_nd.strides


def check_same_memory(original, view):
    assert original._handle.ptr() == view._handle.ptr()

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

    @unittest.skipIf(not train.cuda().enabled(), "NO GPU")
    def test_case4(self):
        shape = (8, 8)
        _A = np.random.randint(low=0, high=10, size=shape)
        A = nd.array(_A, device=train.cuda())
        lhs = A[4:, 4:].compact()
        assert lhs.is_compact(), "array is not compact"
        rhs = _A[4:, 4:]
        np.testing.assert_allclose(lhs.numpy(), rhs, atol=1e-5)

    @unittest.skipIf(not train.cuda().enabled(), "NO GPU")
    def test_case5(self):
        shape = (8, 8, 2, 2, 2, 2)
        _A = np.random.randint(low=0, high=10, size=shape)
        A = nd.array(_A, device=train.cuda())
        lhs = A[1:3, 5:8, 1:2, 0:1, 0:1, 1:2].compact()
        assert lhs.is_compact(), "array is not compact"
        rhs = _A[1:3, 5:8, 1:2, 0:1, 0:1, 1:2]
        np.testing.assert_allclose(lhs.numpy(), rhs, atol=1e-5)
        
    @unittest.skipIf(not train.cuda().enabled(), "NO GPU")
    def test_case6(self):
        shape = (7, 8)
        _A = np.random.randint(low=0, high=10, size=shape)
        A = nd.array(_A, device=train.cuda())
        lhs = A.permute((1,0))[3:7,2:5].compact()
        assert lhs.is_compact(), "array is not compact"
        rhs = _A.transpose()[3:7,2:5]
        np.testing.assert_allclose(lhs.numpy(), rhs, atol=1e-5)

    def test_getitem_cpu(self):
        getitem_params = [
            {"shape": (8, 16), "fn": lambda X: X[3:4, 3:4]},
            {"shape": (8, 16), "fn": lambda X: X[1:2, 1:3]},
            {"shape": (8, 16), "fn": lambda X: X[3:4, 1:4]},
            {"shape": (8, 16), "fn": lambda X: X[1:4, 3:4]},
        ]
        for params in getitem_params:
            shape = params['shape']
            fn = params['fn']
            _A = np.random.randn(5, 5)
            A = nd.array(_A, device=nd.cpu())
            lhs = fn(_A)
            rhs = fn(A)
            np.testing.assert_allclose(lhs, rhs.numpy(), atol=1e-5, rtol=1e-5)
            compare_strides(lhs, rhs)
            check_same_memory(A, rhs)

    @unittest.skipIf(not train.cuda().enabled(), "NO GPU")
    def test_getitem_cuda(self):
        getitem_params = [
            {"shape": (8, 16), "fn": lambda X: X[3:4, 3:4]},
            {"shape": (8, 16), "fn": lambda X: X[1:2, 1:3]},
            {"shape": (8, 16), "fn": lambda X: X[3:4, 1:4]},
            {"shape": (8, 16), "fn": lambda X: X[1:4, 3:4]},
        ]
        for params in getitem_params:
            shape = params['shape']
            fn = params['fn']
            _A = np.random.randn(5, 5)
            A = nd.array(_A, device=train.cuda())
            lhs = fn(_A)
            rhs = fn(A)
            np.testing.assert_allclose(lhs, rhs.numpy(), atol=1e-5, rtol=1e-5)
            compare_strides(lhs, rhs)
            check_same_memory(A, rhs)

if __name__ == '__main__':
    unittest.main()