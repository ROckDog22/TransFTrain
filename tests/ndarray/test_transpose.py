import unittest

import sys
sys.path.append('./python')
import TransFTrain as train
import TransFTrain.backend_ndarray as nd
import numpy as np
import torch

TRANSPOSE_SHAPES = [(1, 1, 1), (4, 5, 6)]
TRANSPOSE_AXES = [(0, 1), (0, 2), None]

np.random.seed(1)

class TestTranspose(unittest.TestCase):
    def test_transpose_cpu(self):
        device = train.cpu()
        for shape in TRANSPOSE_SHAPES:
            for axes in TRANSPOSE_AXES:
                _A = np.random.randn(*shape).astype(np.float32)
                A = train.Tensor(nd.array(_A), device=device)
                if axes is None:
                    np_axes = (_A.ndim - 2, _A.ndim - 1)
                else:
                    np_axes = axes
                np.testing.assert_allclose(np.swapaxes(_A, np_axes[0], np_axes[1]), train.transpose(A, axes=axes).numpy(), atol=1e-5, rtol=1e-5)

    @unittest.skipIf(not train.cuda().enabled(), "NO GPU")
    def test_transpose_cuda(self):
        device = train.cuda()
        for shape in TRANSPOSE_SHAPES:
            for axes in TRANSPOSE_AXES:
                _A = np.random.randn(*shape).astype(np.float32)
                A = train.Tensor(nd.array(_A), device=device)
                if axes is None:
                    np_axes = (_A.ndim - 2, _A.ndim - 1)
                else:
                    np_axes = axes
                np.testing.assert_allclose(np.swapaxes(_A, np_axes[0], np_axes[1]), train.transpose(A, axes=axes).numpy(), atol=1e-5, rtol=1e-5)

if __name__=="__main__":
    unittest.main()