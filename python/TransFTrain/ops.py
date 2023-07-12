"""Operator and gradient implementations."""
from numbers import Number
from typing import Optional, List, Tuple, Union
from itertools import zip_longest
from functools import reduce
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
import numpy

from .backend_selection import array_api, NDArray, default_device


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple([out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        mid = a*b
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad * self.scalar


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a**self.scalar

    def gradient(self, out_grad, node):
        return out_grad * self.scalar * power_scalar(node.inputs[0], self.scalar-1)        


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        return a / b

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        return out_grad / rhs, - out_grad * lhs / power_scalar(rhs, 2) 

def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        # 有问题 后面要改
        return a / self.scalar
        # return (a / self.scalar).astype(a.dtype)

    def gradient(self, out_grad, node):
        return out_grad / self.scalar


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        n = a.ndim
        axis1, axis2 = self.axes if self.axes else (n-2, n-1)
        axes = list(range(n))
        axes[axis1] = axis2
        axes[axis2] = axis1
        return array_api.permute(a, axes)

    def gradient(self, out_grad, node):
        return out_grad.transpose(self.axes)


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.reshape(a, self.shape)

    def gradient(self, out_grad, node):
        return out_grad.reshape(node.inputs[0].shape)


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        shapes = tuple(zip_longest(reversed(node.inputs[0].shape), reversed(out_grad.shape)))
        axes = tuple(i for i, (d1, d2) in enumerate(shapes[::-1]) if d1!=d2)
        return summation(out_grad, axes=axes).reshape(node.inputs[0].shape)

def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)

class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = (axes,) if isinstance(axes, int) else axes

    def compute(self, a):
        if self.axes is None:
            return array_api.summation(a)
        return reduce(array_api.summation, reversed(self.axes), a)

    def gradient(self, out_grad, node):
        shape = restore_shape(node.inputs[0], self.axes)
        return out_grad.reshape(shape).broadcast_to(node.inputs[0].shape)

def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        return a @ b

    def gradient(self, out_grad, node):
        lhm, rhm = node.inputs
        dl = matmul(out_grad, rhm.transpose())
        dr = matmul(lhm.transpose(), out_grad)
        dl = summation(dl, axes = tuple(range(out_grad.ndim - lhm.ndim)))
        dr = summation(dr, axes = tuple(range(out_grad.ndim - rhm.ndim)))
        return dl, dr


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        return -a

    def gradient(self, out_grad, node):
        return -out_grad

def negate(a):
    return Negate()(a)

class Abs(TensorOp):
    def compute(self, a):
        return array_api.abs(a)

    def gradient(self, out_grad, node):
        raise NotImplementedError()
    
def abs(a):
    return Abs()(a)


class Log(TensorOp):
    def compute(self, a):
        return array_api.log(a)

    def gradient(self, out_grad, node):
        return out_grad / node.inputs[0]


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return array_api.exp(a)

    def gradient(self, out_grad, node):
        return out_grad * node.realize_cached_data()


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a: NDArray):
        return a.maximum(0)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return multiply(Tensor(node.realize_cached_data() > 0), out_grad)

def relu(a):
    return ReLU()(a)

class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        shape = restore_shape(Z, self.axes)
        max_z = array_api.max(Z, self.axes, keepdims=True)
        max_z_ = array_api.broadcast_to(array_api.reshape(max_z, shape), Z.shape)
        ret = array_api.log(array_api.exp(Z - max_z_).sum(self.axes))
        return ret + max_z.reshape(ret.shape)

    def gradient(self, out_grad, node):
        # Tensor compute
        inp = node.inputs[0]
        shape = restore_shape(inp, self.axes)
        return out_grad.reshape(shape).broadcast_to(inp.shape) * exp(inp-node.reshape(shape).broadcast_to(inp.shape))


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)


class Tanh(TensorOp):
    def compute(self, a):
        return array_api.tanh(a)

    def gradient(self, out_grad, node):
        return out_grad * (1.0 - tanh(node.inputs[0]) ** 2)

def tanh(a):
    return Tanh()(a)



# todo 这里需要完成
class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args):
        n = len(args)
        m = args[0].size
        axes = [d + 1 for d in range(args[0].ndim)]
        axes.insert(self.axis, 0)

        ret = array_api.empty((n, m), dtype=args[0].dtype, device=args[0].device)
        for i, a in enumerate(args):
            ret[i, :] = a.compact().reshape((1, m))
        return ret.reshape((n, *args[0].shape)).permute(axes)

    def gradient(self, out_grad, node):
        return split(out_grad, axis=self.axis)


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))

# todo 这里需要完成
class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        n = A.shape[self.axis]
        m = A.size // n
        axes = [self.axis] + [d for d in range(A.ndim) if d != self.axis]
        shape = [d for i, d in enumerate(A.shape) if i != self.axis]

        ret = A.permute(axes).compact().reshape((n, m))
        return [array_api.array(ret[i, :]).reshape(shape) for i in range(n)]

    def gradient(self, out_grad, node):
        return stack(out_grad, axis=self.axis)


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self)

# additional helper functions
def full(
    shape, fill_value, *, rand={}, dtype="float32", device=None, requires_grad=False
):
    # numpy do not need device argument
    kwargs = {"device": device} if array_api is not numpy else {}
    device = device if device else default_device()

    if not rand or "dist" not in rand:
        arr = array_api.full(shape, fill_value, dtype=dtype, **kwargs)
    else:
        if rand["dist"] == "normal":
            arr = array_api.randn(
                shape, dtype, mean=rand["mean"], std=rand["std"], **kwargs
            )
        if rand["dist"] == "binomial":
            arr = array_api.randb(
                shape, dtype, ntrials=rand["trials"], p=rand["prob"], **kwargs
            )
        if rand["dist"] == "uniform":
            arr = array_api.randu(
                shape, dtype, low=rand["low"], high=rand["high"], **kwargs
            )

    return Tensor.make_const(arr, requires_grad=requires_grad)


def zeros(shape, *, dtype="float32", device=None, requires_grad=False):
    return full(shape, 0, dtype=dtype, device=device, requires_grad=requires_grad)


def randn(
    shape, *, mean=0.0, std=1.0, dtype="float32", device=None, requires_grad=False
):
    return full(
        shape,
        0,
        rand={"dist": "normal", "mean": mean, "std": std},
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )


def randb(shape, *, n=1, p=0.5, dtype="float32", device=None, requires_grad=False):
    return full(
        shape,
        0,
        rand={"dist": "binomial", "trials": n, "prob": p},
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )


def randu(shape, *, low=0, high=1, dtype="float32", device=None, requires_grad=False):
    return full(
        shape,
        0,
        rand={"dist": "uniform", "low": low, "high": high},
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )


def zeros_like(array, *, device=None, requires_grad=False):
    device = device if device else array.device
    return full(
        array.shape, 0, dtype=array.dtype, device=device, requires_grad=requires_grad
    )


def ones_like(array, *, device=None, requires_grad=False):
    device = device if device else array.device
    return full(
        array.shape, 1, dtype=array.dtype, device=device, requires_grad=requires_grad
    )


def restore_shape(original, axes=None):
    if isinstance(axes, int):
        axes = axes,
    shape = [1] * len(original.shape)
    if axes:
        shape = list(original.shape)
        for i in axes:
            shape[i] = 1
    return tuple(shape)