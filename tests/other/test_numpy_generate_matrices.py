import sys
sys.path.append("./python")
import numpy as np
import TransFTrain as train
import TransFTrain.nn as nn

import unittest

class TestnnAndoptim(unittest.TestCase):
    """Deterministically generate a matrix"""
    def get_tensor(self, *shape, entropy=1):
        np.random.seed(np.prod(shape) * len(shape) * entropy)
        return train.Tensor(np.random.randint(0, 100, size=shape) / 20, dtype="float32")

    def check_prng(self, *shape):
        """ We want to ensure that numpy generates random matrices on your machine/colab
            Such that our tests will make sense
            So this matrix should match our to full precision
        """
        return self.get_tensor(*shape).cached_data

    def test_check_prng_contact_us_if_this_fails_1(self):
        np.testing.assert_allclose(self.check_prng(3, 3),
            np.array([[2.1 , 0.95, 3.45],
            [3.1 , 2.45, 2.3 ],
            [3.3 , 0.4 , 1.2 ]], dtype=np.float32), rtol=1e-08, atol=1e-08)

if "__main__" == __name__:
    unittest.main()
