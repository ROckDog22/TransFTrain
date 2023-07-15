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

BROADCAST_SHAPES = [((1, 1, 1), (3, 3, 3)),
    ((4, 1, 6), (4, 3, 6))]
class TestBroadcast_to(unittest.TestCase):
    def setUp(self):
        # 1 1 2 3
        self.x = np.array([[[[1,2,3],[1,2,3]]]])
        self.x1 = np.array([[[[1,2,3],[1,2,3]]], [[[1,2,3],[1,2,3]]]])

    def test_case1(self):
        x = nd.NDArray(self.x)
        z = x.broadcast_to((1,2,2,3))
        self.assertEqual(x.strides, (6, 6, 3, 1))
        self.assertEqual(x.shape, (1, 1, 2, 3))
        self.assertEqual(z.strides, (0, 0, 3, 1))
        self.assertEqual(z.shape, (1, 2, 2, 3))

    def test_case2(self):
        x = nd.NDArray(self.x1)
        z = x.broadcast_to((2,2,2,3))
        self.assertEqual(x.strides, (6, 6, 3, 1))
        self.assertEqual(x.shape, (2, 1, 2, 3))
        self.assertEqual(z.strides, (6, 0, 3, 1))
        self.assertEqual(z.shape, (2, 2, 2, 3))

    def test_case3(self):
        shape = (4, 1, 4)
        _A = np.random.randint(low=0, high=10, size=shape)
        A = nd.array(_A, device=nd.cpu())
        lhs = A.broadcast_to((4,5,4)).compact()
        assert lhs.is_compact(), "array is not compact"
        rhs = np.broadcast_to(_A, shape=(4, 5, 4))
        np.testing.assert_allclose(lhs.numpy(), rhs, atol=1e-5)


    @unittest.skipIf(not train.cuda().enabled(), "NO GPU")
    def test_case4(self):
        shape = (4, 1, 4)
        _A = np.random.randint(low=0, high=10, size=shape)
        A = nd.array(_A, device=train.cuda())
        lhs = A.broadcast_to((4,5,4))
        lhs = lhs.compact()
        assert lhs.is_compact(), "array is not compact"
        rhs = np.broadcast_to(_A, shape=(4, 5, 4))
        np.testing.assert_allclose(lhs.numpy(), rhs, atol=1e-5)



    def test_broadcast_to_cpu(self):
        broadcast_params = [
            {"from_shape": (1, 3, 4), "to_shape": (6, 3, 4)},
        ]
        for params in broadcast_params:
            from_shape, to_shape = params['from_shape'], params['to_shape']
            _A = np.random.randn(*from_shape)
            A = nd.array(_A, device=nd.cpu())
            lhs = np.broadcast_to(_A, shape=to_shape)
            rhs = A.broadcast_to(to_shape)
            np.testing.assert_allclose(lhs, rhs.numpy(), atol=1e-5, rtol=1e-5)
            compare_strides(lhs, rhs)
            check_same_memory(A, rhs)

    @unittest.skipIf(not train.cuda().enabled(), "NO GPU")
    def test_broadcast_to_cuda(self):
        broadcast_params = [
            {"from_shape": (1, 3, 4), "to_shape": (6, 3, 4)},
        ]
        for params in broadcast_params:
            from_shape, to_shape = params['from_shape'], params['to_shape']
            _A = np.random.randn(*from_shape)
            A = nd.array(_A, device=train.cuda())
            lhs = np.broadcast_to(_A, shape=to_shape)
            rhs = A.broadcast_to(to_shape)
            np.testing.assert_allclose(lhs, rhs.numpy(), atol=1e-5, rtol=1e-5)
            compare_strides(lhs, rhs)
            check_same_memory(A, rhs)


    def test_broadcast_to_cpu(self):
        device = train.cpu()
        for shape, shape_to in BROADCAST_SHAPES:
            _A = np.random.randn(*shape).astype(np.float32)
            A = train.Tensor(nd.array(_A), device=device)
            np.testing.assert_allclose(np.broadcast_to(_A, shape_to), train.broadcast_to(A, shape_to).numpy(), atol=1e-5, rtol=1e-5)

    @unittest.skipIf(not train.cuda().enabled(), "NO GPU")
    def test_broadcast_to_cuda(self):
        device = train.cuda()
        for shape, shape_to in BROADCAST_SHAPES:
            _A = np.random.randn(*shape).astype(np.float32)
            A = train.Tensor(nd.array(_A), device=device)
            np.testing.assert_allclose(np.broadcast_to(_A, shape_to), train.broadcast_to(A, shape_to).numpy(), atol=1e-5, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()