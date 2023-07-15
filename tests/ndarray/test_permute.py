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


class TestPermute(unittest.TestCase):
    def setUp(self):
        # 1 1 2 3
        self.x = np.array([[[[1,2,3],[1,2,3]]]])

    def test_case1(self):
        x = nd.NDArray(self.x)
        z = x.permute((0,3,1,2))
        self.assertEqual(x.strides, (6, 6, 3, 1))
        self.assertEqual(x.shape, (1, 1, 2, 3))
        self.assertEqual(z.strides, (6, 1, 6, 3))
        self.assertEqual(z.shape, (1, 3, 1, 2))

    def test_case2(self):
        shape = (4, 4)
        _A = np.random.randint(low=0, high=10, size=shape)
        A = nd.array(_A, device=nd.cpu())
        lhs = A.permute((1,0)).compact()
        assert lhs.is_compact(), "array is not compact"
        rhs = _A.transpose()
        np.testing.assert_allclose(lhs.numpy(), rhs, atol=1e-5)

    def test_case3(self):
        shape = (4, 4)
        _A = np.random.randint(low=0, high=10, size=shape)
        A = nd.array(_A, device=train.cuda())
        lhs = A.permute((1,0)).compact()
        assert lhs.is_compact(), "array is not compact"
        rhs = _A.transpose()
        np.testing.assert_allclose(lhs.numpy(), rhs, atol=1e-5)

    def test_permute_cpu(self):
        permute_params = [
            {"dims": (4, 5, 6), "axes": (0, 1, 2)},
            {"dims": (4, 5, 6), "axes": (1, 0, 2)},
            {"dims": (4, 5, 6), "axes": (2, 1, 0)},
        ]       
        for params in permute_params:
            dims = params['dims']
            axes = params['axes']
            _A = np.random.randn(*dims)
            A = nd.array(_A, device=nd.cpu())
            lhs = np.transpose(_A, axes=axes)
            rhs = A.permute(axes)
            np.testing.assert_allclose(lhs, rhs.numpy(), atol=1e-5, rtol=1e-5)
            compare_strides(lhs, rhs)
            check_same_memory(A, rhs)


    @unittest.skipIf(not train.cuda().enabled(), "NO GPU")
    def test_permute_cuda(self):
        permute_params = [
            {"dims": (4, 5, 6), "axes": (0, 1, 2)},
            {"dims": (4, 5, 6), "axes": (1, 0, 2)},
            {"dims": (4, 5, 6), "axes": (2, 1, 0)},
        ]       
        for params in permute_params:
            dims = params['dims']
            axes = params['axes']
            _A = np.random.randn(*dims)
            A = nd.array(_A, device=train.cuda())
            lhs = np.transpose(_A, axes=axes)
            rhs = A.permute(axes)
            np.testing.assert_allclose(lhs, rhs.numpy(), atol=1e-5, rtol=1e-5)
            compare_strides(lhs, rhs)
            check_same_memory(A, rhs)

    def test_permute_cpu(self):
        permute_params = [
        {"dims": (4, 5, 6), "axes": (0, 1, 2)},
        {"dims": (4, 5, 6), "axes": (1, 0, 2)},
        {"dims": (4, 5, 6), "axes": (2, 1, 0)},
        ]
        for params in permute_params:
            dims = params['dims']
            axes = params['axes']
            _A = np.random.randn(*dims)
            A = nd.array(_A, device=nd.cpu())
            lhs = np.transpose(_A, axes=axes)
            rhs = A.permute(axes)
            np.testing.assert_allclose(lhs, rhs.numpy(), atol=1e-5, rtol=1e-5)
            compare_strides(lhs, rhs)
            check_same_memory(A, rhs)

    @unittest.skipIf(not train.cuda().enabled(), "NO GPU")
    def test_permute_cuda(self):
        permute_params = [
        {"dims": (4, 5, 6), "axes": (0, 1, 2)},
        {"dims": (4, 5, 6), "axes": (1, 0, 2)},
        {"dims": (4, 5, 6), "axes": (2, 1, 0)},
        ]
        for params in permute_params:
            dims = params['dims']
            axes = params['axes']
            _A = np.random.randn(*dims)
            A = nd.array(_A, device=train.cuda())
            lhs = np.transpose(_A, axes=axes)
            rhs = A.permute(axes)
            np.testing.assert_allclose(lhs, rhs.numpy(), atol=1e-5, rtol=1e-5)
            compare_strides(lhs, rhs)
            check_same_memory(A, rhs)
    
if __name__ == '__main__':
    unittest.main()