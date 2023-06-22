import sys
sys.path.append("./python")
import numpy as np
import TransFTrain as train
import TransFTrain.nn as nn

sys.path.append("./apps")
from mlp_resnet import *
import unittest

class TestnnAndoptim(unittest.TestCase):
    """Deterministically generate a matrix"""
    def get_tensor(*shape, entropy=1):
        np.random.seed(np.prod(shape) * len(shape) * entropy)
        return train.Tensor(np.random.randint(0, 100, size=shape) / 20, dtype="float32")

    def get_int_tensor(*shape, low=0, high=10, entropy=1):
        np.random.seed(np.prod(shape) * len(shape) * entropy)
        return train.Tensor(np.random.randint(low, high, size=shape))

    def check_prng(*shape):
        """ We want to ensure that numpy generates random matrices on your machine/colab
            Such that our tests will make sense
            So this matrix should match our to full precision
        """
        return get_tensor(*shape).cached_data


    def learn_model_1d(feature_size, nclasses, _model, optimizer, epochs=1, **kwargs):
        np.random.seed(42)
        model = _model([])
        X = get_tensor(1024, feature_size).cached_data
        y = get_int_tensor(1024, low=0, high=nclasses).cached_data.astype(np.uint8)
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

    def learn_model_1d_eval(feature_size, nclasses, _model, optimizer, epochs=1, **kwargs):
        np.random.seed(42)
        model = _model([])
        X = get_tensor(1024, feature_size).cached_data
        y = get_int_tensor(1024, low=0, high=nclasses).cached_data.astype(np.uint8)
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

        X_test = train.Tensor(get_tensor(batch, feature_size).cached_data)
        y_test = train.Tensor(get_int_tensor(batch, low=0, high=nclasses).cached_data.astype(np.uint8))

        model.eval()

        return np.array(loss_func(model(X_test), y_test).cached_data)

    def init_a_tensor_of_shape(shape, init_fn):
        x = get_tensor(*shape)
        np.random.seed(42)
        init_fn(x)
        return x.cached_data

    def global_tensor_count():
        return np.array(train.autograd.TENSOR_COUNTER)

    def nn_linear_weight_init():
        np.random.seed(1337)
        f = train.nn.Linear(7, 4)
        f.weight.cached_data
        return f.weight.cached_data

    def nn_linear_bias_init():
        np.random.seed(1337)
        f = train.nn.Linear(7, 4)
        return f.bias.cached_data

    class UselessModule(train.nn.Module):
        def __init__(self):
            super().__init__()
            self.stuff = {'layer1': nn.Linear(4, 4),
                        'layer2': [nn.Dropout(0.1), nn.Sequential(nn.Linear(4, 4))]}

        def forward(self, x):
            raise NotImplementedError()

    def check_training_mode():
        model = nn.Sequential(
            nn.BatchNorm1d(4),
            nn.Sequential(
                nn.LayerNorm1d(4),
                nn.Linear(4, 4),
                nn.Dropout(0.1),
            ),
            nn.Linear(4, 4),
            UselessModule()
        )

        model_refs = [
            model.modules[0],
            model.modules[1].modules[0],
            model.modules[1].modules[1],
            model.modules[1].modules[2],
            model.modules[2],
            model.modules[3],
            model.modules[3].stuff['layer1'],
            model.modules[3].stuff['layer2'][0],
            model.modules[3].stuff['layer2'][1].modules[0]
        ]

        eval_mode = [1 if not x.training else 0 for x in model_refs]
        model.eval()
        eval_mode.extend([1 if not x.training else 0 for x in model_refs])
        model.train()
        eval_mode.extend([1 if not x.training else 0 for x in model_refs])

        return np.array(eval_mode)

    def power_scalar_forward(shape, power=2):
        x = get_tensor(*shape)
        return (x**power).cached_data

    def power_scalar_backward(shape, power=2):
        x = get_tensor(*shape)
        y = (x**power).sum()
        y.backward()
        return x.grad.cached_data


    def dropout_forward(shape, prob=0.5):
        np.random.seed(3)
        x = get_tensor(*shape)
        f = nn.Dropout(prob)
        return f(x).cached_data

    def dropout_backward(shape, prob=0.5):
        np.random.seed(3)
        x = get_tensor(*shape)
        f = nn.Dropout(prob)
        y = f(x).sum()
        y.backward()
        return x.grad.cached_data

    def num_params(model):
        return np.sum([np.prod(x.shape) for x in model.parameters()])

    def residual_block_num_params(dim, hidden_dim, norm):
        model = ResidualBlock(dim, hidden_dim, norm)
        return np.array(num_params(model))

    def residual_block_forward(dim, hidden_dim, norm, drop_prob):
        np.random.seed(2)
        input_tensor = train.Tensor(np.random.randn(1, dim))
        output_tensor = ResidualBlock(dim, hidden_dim, norm, drop_prob)(input_tensor)
        return output_tensor.numpy()

    def mlp_resnet_num_params(dim, hidden_dim, num_blocks, num_classes, norm):
        model = MLPResNet(dim, hidden_dim, num_blocks, num_classes, norm)
        return np.array(num_params(model))

    def mlp_resnet_forward(dim, hidden_dim, num_blocks, num_classes, norm, drop_prob):
        np.random.seed(4)
        input_tensor = train.Tensor(np.random.randn(2, dim), dtype=np.float32)
        output_tensor = MLPResNet(dim, hidden_dim, num_blocks, num_classes, norm, drop_prob)(input_tensor)
        return output_tensor.numpy()

    def train_epoch_1(hidden_dim, batch_size, optimizer, **kwargs):
        np.random.seed(1)
        train_dataset = train.data.MNISTDataset(\
                "./data/train-images-idx3-ubyte.gz",
                "./data/train-labels-idx1-ubyte.gz")
        train_dataloader = train.data.DataLoader(\
                dataset=train_dataset,
                batch_size=batch_size)

        model = MLPResNet(784, hidden_dim)
        opt = optimizer(model.parameters(), **kwargs)
        model.eval()
        return np.array(epoch(train_dataloader, model, opt))

    def eval_epoch_1(hidden_dim, batch_size):
        np.random.seed(1)
        test_dataset = train.data.MNISTDataset(\
                "./data/t10k-images-idx3-ubyte.gz",
                "./data/t10k-labels-idx1-ubyte.gz")
        test_dataloader = train.data.DataLoader(\
                dataset=test_dataset,
                batch_size=batch_size,
                shuffle=False)

        model = MLPResNet(784, hidden_dim)
        model.train()
        return np.array(epoch(test_dataloader, model))

    def train_mnist_1(batch_size, epochs, optimizer, lr, weight_decay, hidden_dim):
        np.random.seed(1)
        out = train_mnist(batch_size, epochs, optimizer, lr, weight_decay, hidden_dim, data_dir="./data")
        return np.array(out)


    def test_check_prng_contact_us_if_this_fails_1():
        np.testing.assert_allclose(check_prng(3, 3),
            np.array([[2.1 , 0.95, 3.45],
            [3.1 , 2.45, 2.3 ],
            [3.3 , 0.4 , 1.2 ]], dtype=np.float32), rtol=1e-08, atol=1e-08)


    def test_op_power_scalar_forward_1():
        np.testing.assert_allclose(power_scalar_forward((2,2), power=2),
            np.array([[11.222499, 17.639997],
            [ 0.0625 , 20.25 ]], dtype=np.float32), rtol=1e-5, atol=1e-5)

    def test_op_power_scalar_forward_2():
        np.testing.assert_allclose(power_scalar_forward((2,2), power=-1.5),
            np.array([[0.16309206, 0.11617859],
            [8. , 0.10475656]], dtype=np.float32), rtol=1e-5, atol=1e-5)

    def test_op_power_scalar_backward_1():
        np.testing.assert_allclose(power_scalar_backward((2,2), power=2),
            np.array([[6.7, 8.4],
            [0.5, 9. ]], dtype=np.float32), rtol=1e-5, atol=1e-5)


    def test_init_kaiming_uniform():
        np.random.seed(42)
        np.testing.assert_allclose(train.init.kaiming_uniform(3,5).numpy(),
            np.array([[-0.35485414, 1.2748126 , 0.65617794, 0.27904832, -0.9729262 ],
            [-0.97299445, -1.2499284 , 1.0357026 , 0.28599644, 0.58851814],
            [-1.3559918 , 1.3291057 , 0.9402898 , -0.81362784, -0.8999349 ]],
            dtype=np.float32), rtol=1e-4, atol=1e-4)

    def test_init_kaiming_normal():
        np.random.seed(42)
        np.testing.assert_allclose(train.init.kaiming_normal(3,5).numpy(),
            np.array([[ 0.4055654 , -0.11289233, 0.5288355 , 1.2435486 , -0.19118543],
            [-0.19117202, 1.2894219 , 0.62660784, -0.38332424, 0.4429984 ],
            [-0.37837896, -0.38026676, 0.19756137, -1.5621868 , -1.4083896 ]],
            dtype=np.float32), rtol=1e-4, atol=1e-4)


    def test_init_xavier_uniform():
        np.random.seed(42)
        np.testing.assert_allclose(train.init.xavier_uniform(3, 5, gain=1.5).numpy(),
            np.array([[-0.32595432, 1.1709901 , 0.60273796, 0.25632226, -0.8936898 ],
            [-0.89375246, -1.1481324 , 0.95135355, 0.26270452, 0.54058844],
            [-1.245558 , 1.2208616 , 0.8637113 , -0.74736494, -0.826643 ]],
            dtype=np.float32), rtol=1e-4, atol=1e-4)

    def test_init_xavier_normal():
        np.random.seed(42)
        np.testing.assert_allclose(train.init.xavier_normal(3, 5, gain=0.33).numpy(),
            np.array([[ 0.08195783 , -0.022813609, 0.10686861 , 0.25129992 ,
            -0.038635306],
            [-0.038632598, 0.2605701 , 0.12662673 , -0.07746328 ,
            0.08952241 ],
            [-0.07646392 , -0.07684541 , 0.039923776, -0.31569123 ,
            -0.28461143 ]], dtype=np.float32), rtol=1e-4, atol=1e-4)
