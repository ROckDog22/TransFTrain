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

""" For converting slice notation to slice objects to make some proceeding tests easier to read """
class _ShapeAndSlices(nd.NDArray):
    def __getitem__(self, idxs):
        idxs = tuple([self.process_slice(s, i) if isinstance(s, slice) else slice(s, s + 1, 1)
                for i, s in enumerate(idxs)])
        return self.shape, idxs
    
ShapeAndSlices = lambda *shape: _ShapeAndSlices(np.ones(shape))

class TestSetitem(unittest.TestCase):
    def test_setitem_ewise_case1_cpu(self):
        device = nd.cpu()
        lhs_shape, lhs_slices = ShapeAndSlices(4, 5, 6)[1:2, 0, 0]
        rhs_shape, rhs_slices = ShapeAndSlices(7, 7, 7)[1:2, 0, 0]
        _A = np.random.randn(*lhs_shape)
        _B = np.random.randn(*rhs_shape)
        A = nd.array(_A, device=device)
        B = nd.array(_B, device=device)
        start_ptr = A._handle.ptr()
        A[lhs_slices] = B[rhs_slices]
        _A[lhs_slices] = _B[rhs_slices]
        end_ptr = A._handle.ptr()
        assert start_ptr == end_ptr, "you should modify in-place"
        compare_strides(_A, A)
        np.testing.assert_allclose(A.numpy(), _A, atol=1e-5, rtol=1e-5)

    @unittest.skipIf(not nd.cuda().enabled(), "NO GPU")
    def test_setitem_ewise_case1_cpu(self):
        device = nd.cuda()
        lhs_shape, lhs_slices = ShapeAndSlices(4, 5, 6)[1:2, 0, 0]
        rhs_shape, rhs_slices = ShapeAndSlices(7, 7, 7)[1:2, 0, 0]
        _A = np.random.randn(*lhs_shape)
        _B = np.random.randn(*rhs_shape)
        A = nd.array(_A, device=device)
        B = nd.array(_B, device=device)
        start_ptr = A._handle.ptr()
        A[lhs_slices] = B[rhs_slices]
        _A[lhs_slices] = _B[rhs_slices]
        end_ptr = A._handle.ptr()
        assert start_ptr == end_ptr, "you should modify in-place"
        compare_strides(_A, A)
        np.testing.assert_allclose(A.numpy(), _A, atol=1e-5, rtol=1e-5)

    def test_setitem_ewise_case2_cpu(self):
        device = nd.cpu()
        lhs_shape, lhs_slices = ShapeAndSlices(4, 5, 6)[1:4:2, 0, 0]
        rhs_shape, rhs_slices = ShapeAndSlices(7, 7, 7)[1:3, 0, 0]
        _A = np.random.randn(*lhs_shape)
        _B = np.random.randn(*rhs_shape)
        A = nd.array(_A, device=device)
        B = nd.array(_B, device=device)
        start_ptr = A._handle.ptr()
        A[lhs_slices] = B[rhs_slices]
        _A[lhs_slices] = _B[rhs_slices]
        end_ptr = A._handle.ptr()
        assert start_ptr == end_ptr, "you should modify in-place"
        compare_strides(_A, A)
        np.testing.assert_allclose(A.numpy(), _A, atol=1e-5, rtol=1e-5)

    @unittest.skipIf(not nd.cuda().enabled(), "NO GPU")
    def test_setitem_ewise_case2_cpu(self):
        device = nd.cuda()
        lhs_shape, lhs_slices = ShapeAndSlices(4, 5, 6)[1:4:2, 0, 0]
        rhs_shape, rhs_slices = ShapeAndSlices(7, 7, 7)[1:3, 0, 0]
        _A = np.random.randn(*lhs_shape)
        _B = np.random.randn(*rhs_shape)
        A = nd.array(_A, device=device)
        B = nd.array(_B, device=device)
        start_ptr = A._handle.ptr()
        A[lhs_slices] = B[rhs_slices]
        _A[lhs_slices] = _B[rhs_slices]
        end_ptr = A._handle.ptr()
        assert start_ptr == end_ptr, "you should modify in-place"
        compare_strides(_A, A)
        np.testing.assert_allclose(A.numpy(), _A, atol=1e-5, rtol=1e-5)
    
    @unittest.skipIf(not nd.cuda().enabled(), "NO GPU")
    def test_setitem_ewise_case3_cpu(self):
        device = nd.cpu()
        lhs_shape, lhs_slices = ShapeAndSlices(4, 5, 6)[1:4:2, 0, 0]
        rhs_shape, rhs_slices = ShapeAndSlices(7, 7, 7)[1:3, 0, 0]
        _A = np.random.randn(*lhs_shape)
        _B = np.random.randn(*rhs_shape)
        A = nd.array(_A, device=device)
        B = nd.array(_B, device=device)
        start_ptr = A._handle.ptr()
        A[lhs_slices] = B[rhs_slices]
        _A[lhs_slices] = _B[rhs_slices]
        end_ptr = A._handle.ptr()
        assert start_ptr == end_ptr, "you should modify in-place"
        compare_strides(_A, A)
        np.testing.assert_allclose(A.numpy(), _A, atol=1e-5, rtol=1e-5)
 

    def test_setitem_ewise_case3_cpu(self):
        device = nd.cpu()
        lhs_shape, lhs_slices = ShapeAndSlices(4, 5, 6)[1:3, 2:5, 2:6]
        rhs_shape, rhs_slices = ShapeAndSlices(7, 7, 7)[:2, :3, :4]
        _A = np.random.randn(*lhs_shape)
        _B = np.random.randn(*rhs_shape)
        A = nd.array(_A, device=device)
        B = nd.array(_B, device=device)
        start_ptr = A._handle.ptr()
        A[lhs_slices] = B[rhs_slices]
        _A[lhs_slices] = _B[rhs_slices]
        end_ptr = A._handle.ptr()
        assert start_ptr == end_ptr, "you should modify in-place"
        compare_strides(_A, A)
        np.testing.assert_allclose(A.numpy(), _A, atol=1e-5, rtol=1e-5)

    @unittest.skipIf(not nd.cuda().enabled(), "NO GPU")
    def test_setitem_ewise_case3_cuda(self):
        device = nd.cuda()
        lhs_shape, lhs_slices = ShapeAndSlices(4, 5, 6)[1:3, 2:5, 2:6]
        rhs_shape, rhs_slices = ShapeAndSlices(7, 7, 7)[:2, :3, :4]
        _A = np.random.randn(*lhs_shape)
        _B = np.random.randn(*rhs_shape)
        A = nd.array(_A, device=device)
        B = nd.array(_B, device=device)
        start_ptr = A._handle.ptr()
        A[lhs_slices] = B[rhs_slices]
        _A[lhs_slices] = _B[rhs_slices]
        end_ptr = A._handle.ptr()
        assert start_ptr == end_ptr, "you should modify in-place"
        compare_strides(_A, A)
        np.testing.assert_allclose(A.numpy(), _A, atol=1e-5, rtol=1e-5)

# test_setitem_scalar
    def test_setitem_scalar_case1_cpu(self):
        device = nd.cpu()
        shape, slices = ShapeAndSlices(4, 5, 6)[1,    2,   3]
        _A = np.random.randn(*shape)
        A = nd.array(_A, device=device)
        # probably tear these out using lambdas
        print(slices)
        start_ptr = A._handle.ptr()
        _A[slices] = 4.0
        A[slices] = 4.0
        end_ptr = A._handle.ptr()
        assert start_ptr == end_ptr, "you should modify in-place"
        np.testing.assert_allclose(A.numpy(), _A, atol=1e-5, rtol=1e-5)
        compare_strides(_A, A)

    @unittest.skipIf(not nd.cuda().enabled(), "NO GPU")
    def test_setitem_scalar_case1_cuda(self):
        device = nd.cuda()
        shape, slices = ShapeAndSlices(4, 5, 6)[1,    2,   3]
        _A = np.random.randn(*shape)
        A = nd.array(_A, device=device)
        # probably tear these out using lambdas
        print(slices)
        start_ptr = A._handle.ptr()
        _A[slices] = 4.0
        A[slices] = 4.0
        end_ptr = A._handle.ptr()
        assert start_ptr == end_ptr, "you should modify in-place"
        np.testing.assert_allclose(A.numpy(), _A, atol=1e-5, rtol=1e-5)
        compare_strides(_A, A)


    def test_setitem_scalar_case2_cpu(self):
        device = nd.cpu()
        shape, slices = ShapeAndSlices(4, 5, 6)[1:4,  2,   3]
        _A = np.random.randn(*shape)
        A = nd.array(_A, device=device)
        # probably tear these out using lambdas
        print(slices)
        start_ptr = A._handle.ptr()
        _A[slices] = 4.0
        A[slices] = 4.0
        end_ptr = A._handle.ptr()
        assert start_ptr == end_ptr, "you should modify in-place"
        np.testing.assert_allclose(A.numpy(), _A, atol=1e-5, rtol=1e-5)
        compare_strides(_A, A)

    @unittest.skipIf(not nd.cuda().enabled(), "NO GPU")
    def test_setitem_scalar_case2_cuda(self):
        device = nd.cuda()
        shape, slices = ShapeAndSlices(4, 5, 6)[1:4,  2,   3]
        _A = np.random.randn(*shape)
        A = nd.array(_A, device=device)
        # probably tear these out using lambdas
        print(slices)
        start_ptr = A._handle.ptr()
        _A[slices] = 4.0
        A[slices] = 4.0
        end_ptr = A._handle.ptr()
        assert start_ptr == end_ptr, "you should modify in-place"
        np.testing.assert_allclose(A.numpy(), _A, atol=1e-5, rtol=1e-5)
        compare_strides(_A, A)



    def test_setitem_scalar_case3_cpu(self):
        device = nd.cpu()
        shape, slices = ShapeAndSlices(4, 5, 6)[:4,  2:5, 3]
        _A = np.random.randn(*shape)
        A = nd.array(_A, device=device)
        # probably tear these out using lambdas
        print(slices)
        start_ptr = A._handle.ptr()
        _A[slices] = 4.0
        A[slices] = 4.0
        end_ptr = A._handle.ptr()
        assert start_ptr == end_ptr, "you should modify in-place"
        np.testing.assert_allclose(A.numpy(), _A, atol=1e-5, rtol=1e-5)
        compare_strides(_A, A)


    @unittest.skipIf(not nd.cuda().enabled(), "NO GPU")
    def test_setitem_scalar_case3_cuda(self):
        device = nd.cuda()
        shape, slices = ShapeAndSlices(4, 5, 6)[:4,  2:5, 3]
        _A = np.random.randn(*shape)
        A = nd.array(_A, device=device)
        # probably tear these out using lambdas
        print(slices)
        start_ptr = A._handle.ptr()
        _A[slices] = 4.0
        A[slices] = 4.0
        end_ptr = A._handle.ptr()
        assert start_ptr == end_ptr, "you should modify in-place"
        np.testing.assert_allclose(A.numpy(), _A, atol=1e-5, rtol=1e-5)
        compare_strides(_A, A)

    def test_setitem_scalar_case4_cpu(self):
        device = nd.cpu()
        shape, slices = ShapeAndSlices(4, 5, 6)[1::2, 2:5, ::2]
        _A = np.random.randn(*shape)
        A = nd.array(_A, device=device)
        # probably tear these out using lambdas
        print(slices)
        start_ptr = A._handle.ptr()
        _A[slices] = 4.0
        A[slices] = 4.0
        end_ptr = A._handle.ptr()
        assert start_ptr == end_ptr, "you should modify in-place"
        np.testing.assert_allclose(A.numpy(), _A, atol=1e-5, rtol=1e-5)
        compare_strides(_A, A)


    @unittest.skipIf(not nd.cuda().enabled(), "NO GPU")
    def test_setitem_scalar_case4_cuda(self):
        device = nd.cuda()
        shape, slices = ShapeAndSlices(4, 5, 6)[1::2, 2:5, ::2]
        _A = np.random.randn(*shape)
        A = nd.array(_A, device=device)
        # probably tear these out using lambdas
        print(slices)
        start_ptr = A._handle.ptr()
        _A[slices] = 4.0
        A[slices] = 4.0
        end_ptr = A._handle.ptr()
        assert start_ptr == end_ptr, "you should modify in-place"
        np.testing.assert_allclose(A.numpy(), _A, atol=1e-5, rtol=1e-5)
        compare_strides(_A, A)


if __name__=="__main__":
    unittest.main()

