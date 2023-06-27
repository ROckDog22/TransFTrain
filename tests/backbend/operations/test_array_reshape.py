import unittest
import numpy as np
import sys
sys.path.append('./python')
# 你需要在.vscode里面添加extra地址 才能找到
import TransFTrain as train
from TransFTrain import back


_DEVICES = [nd.cpu(), pytest.param(nd.cuda(), 
    marks=pytest.mark.skipif(not nd.cuda().enabled(), reason="No GPU"))]


def compare_strides(a_np, a_nd):
    size = a_np.itemsize
    assert tuple([x // size for x in a_np.strides]) == a_nd.strides


def check_same_memory(original, view):
    assert original._handle.ptr() == view._handle.ptr()


# TODO test permute, broadcast_to, reshape, getitem, some combinations thereof
@pytest.mark.parametrize("params", [
    {
     "shape": (4, 4),
     "np_fn": lambda X: X.transpose(),
     "nd_fn": lambda X: X.permute((1, 0))
    },
    {
     "shape": (4, 1, 4),
     "np_fn": lambda X: np.broadcast_to(X, shape=(4, 5, 4)),
     "nd_fn": lambda X: X.broadcast_to((4, 5, 4))
    },
    {
     "shape": (4, 3),
     "np_fn": lambda X: X.reshape(2, 2, 3),
     "nd_fn": lambda X: X.reshape((2, 2, 3))
    },
    {
     "shape": (16, 16), # testing for compaction of large ndims array
     "np_fn": lambda X: X.reshape(2, 4, 2, 2, 2, 2, 2),
     "nd_fn": lambda X: X.reshape((2, 4, 2, 2, 2, 2, 2))
    },
    {
     "shape": (2, 4, 2, 2, 2, 2, 2), # testing for compaction of large ndims array
     "np_fn": lambda X: X.reshape(16, 16),
     "nd_fn": lambda X: X.reshape((16, 16))
    },
    {
     "shape": (8, 8),
     "np_fn": lambda X: X[4:,4:],
     "nd_fn": lambda X: X[4:,4:]
    },
    {
     "shape": (8, 8, 2, 2, 2, 2),
     "np_fn": lambda X: X[1:3, 5:8, 1:2, 0:1, 0:1, 1:2],
     "nd_fn": lambda X: X[1:3, 5:8, 1:2, 0:1, 0:1, 1:2]
    }, 
    {
     "shape": (7, 8),
     "np_fn": lambda X: X.transpose()[3:7,2:5],
     "nd_fn": lambda X: X.permute((1, 0))[3:7,2:5]
    },   
], ids=["transpose", "broadcast_to", "reshape1", "reshape2", "reshape3", "getitem1", "getitem2", "transposegetitem"])
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_compact(params, device):
    shape, np_fn, nd_fn = params['shape'], params['np_fn'], params['nd_fn']
    _A = np.random.randint(low=0, high=10, size=shape)
    A = nd.array(_A, device=device)
    
    lhs = nd_fn(A).compact()
    assert lhs.is_compact(), "array is not compact"

    rhs = np_fn(_A)
    np.testing.assert_allclose(lhs.numpy(), rhs, atol=1e-5, rtol=1e-5)

    
class TestArrayReshape(unittest.TestCase):
    def test_case1(self):
        x = train.Tensor([1,2,3], dtype="int8")
        y = 3
        z = train.Tensor([3,6,9], dtype="int8")
        self.assertEqual(z/3, x)

    def test_case2(self):
        x = train.Tensor([1,2,3], dtype="int8")
        y = 3
        z = train.Tensor([3,6,9], dtype="int8")
        self.assertEqual(train.divide_scalar(z,y), x)

if __name__ == '__main__':
    unittest.main()