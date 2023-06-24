"""The module.
"""
from typing import List, Callable, Any
import sys 
sys.path.append('./python')
from TransFTrain.autograd import Tensor
from TransFTrain import ops
import TransFTrain.init as init
import numpy as np
from functools import reduce

class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []



class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(init.kaiming_uniform(
            fan_in=in_features, fan_out=out_features, device=device, dtype=dtype, requires_grad=True
        ))

        self.bias = None

        if bias:
            self.bias = init.kaiming_uniform(
                fan_in = out_features,
                fan_out = 1,
                device = device,
                dtype = dtype,
                requires_grad=True
            )
            self.bias = Parameter(self.bias.reshape((1, out_features)))

    def forward(self, X: Tensor) -> Tensor:
        ret = X @ self.weight
        if self.bias:
            ret += self.bias.broadcast_to(ret.shape)
        return ret


class Flatten(Module):
    def forward(self, X):
        N, _ = X.shape
        return X.reshape((N, -1))


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        return reduce(lambda x,f: f(x), self.modules, x)


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        N, C = logits.shape
        y_one_hot = init.one_hot(C, y, device = y.device)
        return (ops.logsumexp(logits, 1) - (logits * y_one_hot).sum(1)).sum() / N



class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        self.running_mean = Parameter(init.zeros(dim, device=device, dtype=dtype))
        self.running_var = Parameter(init.ones(dim, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        M, N = x.shape
        assert self.dim == N
        if self.training:
            mean = (x.sum(axes=0) / M)
            self.running_mean = (1-self.momentum) * self.running_mean + self.momentum * mean.data
            mean = mean.reshape((1, N)).boradcast_to(x.shape)
            var = ((x-mean)**2).sum(axes=0) / M
            self.running_var = (1-self.momentum) * self.running_var + self.momentum * var.data
            var = var.reshape((1, N)).boradcast_to(x.shape)
        else:
            mean = self.running_mean.reshape((1, M)).broadcast_to(x.shape)
            var = self.running_var.reshape((1, M)).broadcast_to(x.shape)
        x = (x-mean)/ (var+self.eps)**0.5
        weight = self.weight.reshape((1, N)).broadcast_to(x.shape)
        bias = self.bias.reshape((1, N)).broadcast_to(x.shape)
        return weight * x + bias

class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        M, N = x.shape
        assert self.dim == N
        mean = (x.sum(axes=1)/N).reshape((M,1)).broadcast_to(x.shape)
        var = (((x - mean)**2).sum(axes=1)/N).reshape((M,1)).broadcast_to(x.shape)
        y = self.weight.boradcast_to(x.shape) * ((x - mean) / (var + self.eps)**0.5) + self.bias.broadcast_to(x.shape)
        return y
    
class Dropout(Module):
    def __init__(self, p = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            mask = init.randb(*x.shape, p = self.p, dtype="float32", device=x.device)
            x = mask * x / (1 - self.p)
        return x

class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return self.fn(x) + x



