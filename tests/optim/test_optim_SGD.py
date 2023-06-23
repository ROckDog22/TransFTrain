import sys 
sys.path.append('./python')

import TransFTrain as train
import TransFTrain.nn as nn

import unittest
import numpy as np

class TestOptimSGD(unittest.TestCase): 
    def get_tensor(self, *shape, entropy=1):
        np.random.seed(np.prod(shape) * len(shape) * entropy)
        return train.Tensor(np.random.randint(0, 100, size=shape) / 20, dtype="float32")

    def get_int_tensor(self, *shape, low=0, high=10, entropy=1):
        np.random.seed(np.prod(shape) * len(shape) * entropy)
        return train.Tensor(np.random.randint(low, high, size=shape))

    def global_tensor_count(self):
        return np.array(train.autograd.TENSOR_COUNTER)
    
    def learn_model_1d(self, feature_size, nclasses, _model, optimizer, epochs=1, **kwargs):
        np.random.seed(42)
        model = _model([])
        X = self.get_tensor(1024, feature_size).cached_data
        y = self.get_int_tensor(1024, low=0, high=nclasses).cached_data.astype(np.uint8)
        m = X.shape[0]
        batch = 32

        loss_func = nn.SoftmaxLoss()
        opt = optimizer(model.parameters(), **kwargs)

        for _ in range(epochs):
            for i, (X0, y0) in enumerate(zip(np.array_split(X, m//batch), np.array_split(y, m//batch))):
                opt.reset_grad()
                X0, y0 = train.Tensor(X0, dtype="float32"), train.Tensor(y0)
                out = model(X0)
                loss = loss_func(out, y0)
                loss.backward()
                # Opt should not change gradients.
                grad_before = model.parameters()[0].grad.detach().cached_data
                opt.step()
                grad_after = model.parameters()[0].grad.detach().cached_data
                np.testing.assert_allclose(grad_before, grad_after, rtol=1e-5, atol=1e-5, \
                                        err_msg="Optim should not modify gradients in place")


        return np.array(loss.cached_data)

    def learn_model_1d_eval(self, feature_size, nclasses, _model, optimizer, epochs=1, **kwargs):
        np.random.seed(42)
        model = _model([])
        X = self.get_tensor(1024, feature_size).cached_data
        y = self.get_int_tensor(1024, low=0, high=nclasses).cached_data.astype(np.uint8)
        m = X.shape[0]
        batch = 32

        loss_func = nn.SoftmaxLoss()
        opt = optimizer(model.parameters(), **kwargs)

        for i, (X0, y0) in enumerate(zip(np.array_split(X, m//batch), np.array_split(y, m//batch))):
            opt.reset_grad()
            X0, y0 = train.Tensor(X0, dtype="float32"), train.Tensor(y0)
            out = model(X0)
            loss = loss_func(out, y0)
            loss.backward()
            opt.step()

        X_test = train.Tensor(self.get_tensor(batch, feature_size).cached_data)
        y_test = train.Tensor(self.get_int_tensor(batch, low=0, high=nclasses).cached_data.astype(np.uint8))

        model.eval()

        return np.array(loss_func(model(X_test), y_test).cached_data)
    

    def test_optim_sgd_vanilla_1(self):
        np.testing.assert_allclose(self.learn_model_1d(64, 16, lambda z: nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16)), train.optim.SGD, lr=0.01, momentum=0.0),
            np.array(3.207009), rtol=1e-5, atol=1e-5)

    def test_optim_sgd_momentum_1(self):
        np.testing.assert_allclose(self.learn_model_1d(64, 16, lambda z: nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16)), train.optim.SGD, lr=0.01, momentum=0.9),
            np.array(3.311805), rtol=1e-5, atol=1e-5)

    def test_optim_sgd_weight_decay_1(self):
        np.testing.assert_allclose(self.learn_model_1d(64, 16, lambda z: nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16)), train.optim.SGD, lr=0.01, momentum=0.0, weight_decay=0.01),
            np.array(3.202637), rtol=1e-5, atol=1e-5)

    def test_optim_sgd_momentum_weight_decay_1(self):
        np.testing.assert_allclose(self.learn_model_1d(64, 16, lambda z: nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16)), train.optim.SGD, lr=0.01, momentum=0.9, weight_decay=0.01),
            np.array(3.306993), rtol=1e-5, atol=1e-5)

    def test_optim_sgd_layernorm_residual_1(self):
        nn.LayerNorm1d(8)
        np.testing.assert_allclose(self.learn_model_1d(64, 16, lambda z: nn.Sequential(nn.Linear(64, 8), nn.ReLU(), nn.Residual(nn.Linear(8, 8)), nn.Linear(8, 16)), train.optim.SGD, epochs=3, lr=0.01, weight_decay=0.001),
            np.array(2.852236), rtol=1e-5, atol=1e-5)

    # We're checking that you have not allocated too many tensors;
    # if this fails, make sure you're using .detach()/.data whenever possible.
    def test_optim_sgd_z_memory_check_1(self):
        np.testing.assert_allclose(self.global_tensor_count(),
            np.array(387), rtol=1e-5, atol=1000)
        
    
if "__main__" == __name__:
    unittest.main()