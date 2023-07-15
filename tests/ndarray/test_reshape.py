import unittest

import sys
sys.path.append('./python')
# 你需要在.vscode里面添加extra地址 才能找到
import TransFTrain as train
import TransFTrain.backend_ndarray as nd
import numpy as np

RESHAPE_SHAPES = [((1, 1, 1), (1,)),
    ((4, 1, 6), (6, 4, 1))]

def compare_strides(a_np, a_nd):
    size = a_np.itemsize
    assert tuple([x // size for x in a_np.strides]) == a_nd.strides


def check_same_memory(original, view):
    assert original._handle.ptr() == view._handle.ptr()


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
        A = nd.array(_A, device=train.cuda())
        lhs = A.reshape((2, 2, 3)).compact()
        assert lhs.is_compact(), "array is not compact"
        rhs = _A.reshape(2, 2, 3)
        np.testing.assert_allclose(lhs.numpy(), rhs, atol=1e-5)

    def test_case5(self):
        shape = (16, 16)
        _A = np.random.randint(low=0, high=10, size=shape)
        A = nd.array(_A, device=train.cuda())
        lhs = A.reshape((2, 4, 2, 2, 2, 2, 2)).compact()
        assert lhs.is_compact(), "array is not compact"
        rhs = _A.reshape(2, 4, 2, 2, 2, 2, 2)
        np.testing.assert_allclose(lhs.numpy(), rhs, atol=1e-5)

    def test_case6(self):
        shape = (2, 4, 2, 2, 2, 2, 2)
        _A = np.random.randint(low=0, high=10, size=shape)
        A = nd.array(_A, device=train.cuda())
        lhs = A.reshape((16, 16)).compact()
        assert lhs.is_compact(), "array is not compact"
        rhs = _A.reshape(16, 16)
        np.testing.assert_allclose(lhs.numpy(), rhs, atol=1e-5)

    def test_reshape_cpu(self):
        reshape_params = [
            {"shape": (8, 16), "new_shape": (2, 4, 16)},
            {"shape": (8, 16), "new_shape": (8, 4, 2, 2)},
        ]
        for params in reshape_params:
            shape = params['shape']
            new_shape = params['new_shape']
            _A = np.random.randn(*shape)
            A = nd.array(_A, device=nd.cpu())
            lhs = _A.reshape(*new_shape)
            rhs = A.reshape(new_shape)
            np.testing.assert_allclose(lhs, rhs.numpy(), atol=1e-5, rtol=1e-5)
            compare_strides(lhs, rhs)
            check_same_memory(A, rhs)



    def test_reshape_cuda(self):
        reshape_params = [
            {"shape": (8, 16), "new_shape": (2, 4, 16)},
            {"shape": (8, 16), "new_shape": (8, 4, 2, 2)},
        ]
        for params in reshape_params:
            shape = params['shape']
            new_shape = params['new_shape']
            _A = np.random.randn(*shape)
            A = nd.array(_A, device=train.cuda())
            lhs = _A.reshape(*new_shape)
            rhs = A.reshape(new_shape)
            np.testing.assert_allclose(lhs, rhs.numpy(), atol=1e-5, rtol=1e-5)
            compare_strides(lhs, rhs)
            check_same_memory(A, rhs)

    def test_reshape(self):
        device = train.cpu()
        for shape, shape_to in RESHAPE_SHAPES:
            _A = np.random.randn(*shape).astype(np.float32)
            A = train.Tensor(nd.array(_A), device=device)
            np.testing.assert_allclose(np.reshape(_A, shape_to), train.reshape(A, shape_to).numpy(), atol=1e-5, rtol=1e-5)


    @unittest.skipIf(not train.cuda().enabled(), "NO GPU")
    def test_reshape(self):
        device = train.cuda()
        for shape, shape_to in RESHAPE_SHAPES:
            _A = np.random.randn(*shape).astype(np.float32)
            A = train.Tensor(nd.array(_A), device=device)
            np.testing.assert_allclose(np.reshape(_A, shape_to), train.reshape(A, shape_to).numpy(), atol=1e-5, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()