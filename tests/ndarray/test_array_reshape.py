import unittest
import numpy as np
import sys
sys.path.append('../python')
import TransFTrain as train
from TransFTrain import backend_ndarray as nd

params = [
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
        }
    ]

ids=["getitem1", "getitem2", "transposegetitem"]

def compare_strides(a_np, a_nd):
    size = a_np.itemsize
    assert tuple([x // size for x in a_np.strides]) == a_nd.strides

def check_same_memory(original, view):
    assert original._handle.ptr() == view._handle.ptr()

class TestGPUArrayReshape(unittest.TestCase):
# TODO test permute, broadcast_to, reshape, getitem, some combinations thereof    
    @unittest.skip(not nd.cuda().enabled(), "No GPU")
    def test_cuda_compact(self):
        for params in params: 
            shape, np_fn, nd_fn = params['shape'], params['np_fn'], params['nd_fn']
            _A = np.random.randint(low=0, high=10, size=shape)
            A = nd.array(_A, device=nd.cuda())
        
            lhs = nd_fn(A).compact()
            assert lhs.is_compact(), "array is not compact"

            rhs = np_fn(_A)
            np.testing.assert_allclose(lhs.numpy(), rhs, atol=1e-5, rtol=1e-5)

# TODO test permute, broadcast_to, reshape, getitem, some combinations thereof    
    def test_cpu_compact(self):
        for params in params: 
            shape, np_fn, nd_fn = params['shape'], params['np_fn'], params['nd_fn']
            _A = np.random.randint(low=0, high=10, size=shape)
            A = nd.array(_A, device=nd.cpu())
        
            lhs = nd_fn(A).compact()
            assert lhs.is_compact(), "array is not compact"

            rhs = np_fn(_A)
            np.testing.assert_allclose(lhs.numpy(), rhs, atol=1e-5, rtol=1e-5)

if __name__ == '__main__':
    unittest.main()