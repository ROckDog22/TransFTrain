import math
import TransFTrain as train


def rand(*shape, low=0.0, high=1.0, device=None, dtype="float32", requires_grad=False):
    """ Generate random numbers uniform between low and high """
    device = train.default_device() if device is None else device
    array = device.rand(*shape, dtype=dtype) * (high - low) + low
    return train.Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)


def randn(*shape, mean=0.0, std=1.0, device=None, dtype="float32", requires_grad=False):
    """ Generate random normal with specified mean and std deviation """
    device = train.default_device() if device is None else device
    array = device.randn(*shape, dtype=dtype) * std + mean
    return train.Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)


def constant(*shape, c=1.0, device=None, dtype="float32", requires_grad=False):
    """ Generate constant Tensor """
    device = train.default_device() if device is None else device
    array = device.full(shape, c, dtype=dtype)
    return train.Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)


def ones(*shape, device=None, dtype="float32", requires_grad=False):
    """ Generate all-ones Tensor """
    return constant(
        *shape, c=1.0, device=device, dtype=dtype, requires_grad=requires_grad
    )


def zeros(*shape, device=None, dtype="float32", requires_grad=False):
    """ Generate all-zeros Tensor """
    return constant(
        *shape, c=0.0, device=device, dtype=dtype, requires_grad=requires_grad
    )


def randb(*shape, p=0.5, device=None, dtype="bool", requires_grad=False):
    """ Generate binary random Tensor """
    device = train.default_device() if device is None else device
    array = device.rand(*shape) <= p
    return train.Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)


def one_hot(n, i, device=None, dtype="float32", requires_grad=False):
    """ Generate one-hot encoding Tensor """
    device = train.default_device() if device is None else device
    return train.Tensor(
        device.one_hot(n, i.numpy().astype("int32"), dtype=dtype),
        device=device,
        requires_grad=requires_grad,
    )


def zeros_like(array, *, device=None, requires_grad=False):
    device = device if device else array.device
    return zeros(
        *array.shape, dtype=array.dtype, device=device, requires_grad=requires_grad
    )


def ones_like(array, *, device=None, requires_grad=False):
    device = device if device else array.device
    return ones(
        *array.shape, dtype=array.dtype, device=device, requires_grad=requires_grad
    )


def xavier_uniform(fan_in, fan_out, shape=None, gain=1.0, **kwargs):
    alpha = gain * (6 / (fan_in + fan_out)) ** 0.5
    return rand(fan_in, fan_out, low=-alpha, high=alpha, **kwargs)


def xavier_normal(fan_in, fan_out, shape=None, gain=1.0, **kwargs):
    std = gain * (2 / (fan_in + fan_out)) ** 0.5
    return randn(fan_in, fan_out, std=std, **kwargs)


def kaiming_uniform(fan_in, fan_out, shape=None, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    bound = (6 / fan_in) ** 0.5
    if shape is None:
        shape = fan_in, fan_out
    return rand(*shape, low=-bound, high=bound, **kwargs)


def kaiming_normal(fan_in, fan_out, shape=None, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    if shape is None:
        shape = fan_in, fan_out
    return randn(*shape, std=(2 / fan_in) ** 0.5, **kwargs)
