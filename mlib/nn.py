import numpy as np
from typing import Callable
from abc import ABC, abstractmethod
from functools import partial
from typing import Self


def set_seed(seed: int = 0) -> None:
    np.random.seed(seed)


def SimpleStep(lr: float = 3e-4) -> None:
    def __grad__(parameter: np.ndarray, gradient: np.ndarray):
        return parameter - lr * gradient

    return __grad__


def differentiate(f: Callable, x: np.ndarray, h=1e-9) -> np.ndarray:
    z = np.zeros_like(x, dtype=np.float64)
    fx = f(x)

    for idx in np.ndindex(x.shape):
        orig = x[idx]
        x[idx] = orig + h
        z[idx] = (f(x) - fx) / h
        x[idx] = orig

    return z


class ShapeShift:
    def __init__(self, from_shape: tuple, to_shape: tuple):
        self.from_shape = from_shape
        self.to_shape = to_shape

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.from_shape == self.to_shape:
            return x
        return x.reshape(self.to_shape)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        if self.from_shape == self.to_shape:
            return grad
        return grad.reshape(self.from_shape)


class Tensor:
    def __init__(self, value: np.ndarray = None):
        self.history = []
        self.value = value
        self.grad = None

    def push(self, module) -> None:
        self.history.append(module)

    def get(self) -> object:
        return self.history.pop()

    def backward(self) -> None:
        if self.grad is None:
            raise ValueError("Gradient must be set before backward pass.")

        if self.history:
            module = self.get()
            if not isinstance(module, dict):
                self.grad = module.backward(self.grad)
                self.backward()
            else:
                agg_type, histories = next(iter((module.items())))
                if agg_type == "net_add":
                    a_tensor = Tensor(0)
                    a_tensor.grad = self.grad
                    a_tensor.history = histories[0]
                    b_tensor = Tensor(0)
                    b_tensor.grad = self.grad
                    b_tensor.history = histories[1]
                    a_tensor.backward()
                    b_tensor.backward()
                elif agg_type == "net_sub":
                    a_tensor = Tensor(0)
                    a_tensor.grad = self.grad
                    a_tensor.history = histories[0]
                    b_tensor = Tensor(0)
                    b_tensor.grad = -self.grad
                    b_tensor.history = histories[1]
                    a_tensor.backward()
                    b_tensor.backward()
                elif agg_type == "net_split":
                    split_tensor = Tensor(0)
                    split_tensor.grad = self.grad
                    split_tensor.history = histories[0]
                    split_tensor.backward()

    def __add__(self, other):
        add_tensor = Tensor(self.value + other.value)
        add_tensor.history = [{"net_add": (self.history, other.history)}]
        return add_tensor

    def __sub__(self, other):
        sub_tensor = Tensor(self.value - other.value)
        sub_tensor.history = [{"net_sub": (self.history, other.history)}]
        return sub_tensor

    def item(self) -> np.ndarray:
        return self.value

    def flatten(self) -> Self:
        shape_shift = ShapeShift(self.value.shape, (-1, 1))
        self.value = shape_shift.forward(self.value)
        self.push(shape_shift)
        return self

    def reshape(self, new_shape: tuple) -> Self:
        shape_shift = ShapeShift(self.value.shape, new_shape)
        self.value = shape_shift.forward(self.value)
        self.push(shape_shift)
        return self

    def split(self, n: int) -> list[Self]:
        tensor_splits = []
        for _ in range(n):
            new_tensor = Tensor(self.value)
            new_tensor.history = [{"net_split": self.history}]
            tensor_splits.append(new_tensor)
        return tensor_splits


def flatten(x: Tensor) -> Tensor:
    return x.flatten()


def Reshape(new_shape: tuple) -> Tensor:
    def __reshape__(x: Tensor) -> Tensor:
        return x.reshape(new_shape)

    return __reshape__


def Split(n: int) -> Tensor:
    def __split__(x: Tensor) -> list[Tensor]:
        return x.split(n)

    return __split__


class LossFunction(ABC):
    def __init__(self):
        super().__init__()

    def __call__(self, x: Tensor, y: Tensor) -> Tensor:
        """x is model output and y is target output."""
        grad = differentiate(partial(self.loss, y=y.value), x.value)
        loss = Tensor(self.loss(x.value, y.value))
        loss.grad = grad
        loss.history = x.history
        return loss

    @abstractmethod
    def loss(self, x, y) -> np.ndarray:
        """Inputs and outputs are assumed to be of typical numpy arrays."""

        pass


class MSELoss(LossFunction):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        return np.mean((x - y) ** 2)


class CrossEntropyLoss(LossFunction):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        return -(y * np.log(x + 1e-9)).sum()


class Module(ABC):
    def __init__(self):
        super().__init__()

    def __call__(self, x: Tensor) -> Tensor:
        self.forward_cache = None
        self.forward_pass(x.value)
        y = Tensor(self.forward(x.value))
        y.history = x.history
        y.push(self)
        return y

    @abstractmethod
    def forward_pass(self, x: np.ndarray) -> None:
        """Perform the forward pass and cache necessary values."""
        pass

    @abstractmethod
    def backward_pass(self, grad: np.ndarray) -> np.ndarray:
        """Perform the backward pass and return the gradient."""
        pass

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute the forward output."""
        pass

    def backward(self, grad: np.ndarray) -> np.ndarray:
        if self.forward_cache is None:
            raise ValueError("Forward pass must be called before backward pass.")
        return self.backward_pass(grad)

    def zero_grad(self) -> None:
        self.gradients = {key: 0 for key in self.gradients}

    def step(self, optim: Callable) -> None:
        if self.requires_grad:
            for key in self.gradients:
                self.parameters[key] = optim(self.parameters[key], self.gradients[key])

    def card(self) -> dict:
        return {
            "module": self.name,
            "weights": {"parameters": self.parameters, "gradients": self.gradients},
        }


class Linear(Module):
    def __init__(self, n: int, m: int):
        super().__init__()
        self.n = n
        self.m = m
        self.requires_grad = True
        self.name = "Linear"
        self.parameters = {
            "A": np.random.randn(n, m),
            "B": np.random.randn(n, 1),
            "requires_grad": self.requires_grad,
        }
        self.gradients = {"A": 0, "B": 0}
        self.forward_cache = None

    def forward_pass(self, x: np.ndarray) -> None:
        self.forward_cache = {"A": np.tile(x.T, (self.n, 1)), "B": np.ones((self.n, 1))}

    def backward_pass(self, grad: np.ndarray) -> np.ndarray:
        precompute = self.forward_cache["A"] * grad
        if self.requires_grad:
            self.gradients["A"] += precompute
            self.gradients["B"] += self.forward_cache["B"] * grad

        return precompute.sum(axis=0).reshape(-1, 1)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.parameters["A"] @ x + self.parameters["B"]


class ReLU(Module):
    def __init__(self, delta: float = 0, alpha: float = 0, beta: float = 1):
        super().__init__()
        self.alpha = alpha
        self.delta = delta
        self.beta = beta
        self.requires_grad = False
        self.name = "ReLU"
        self.parameters = {
            "delta": delta,
            "alpha": alpha,
            "beta": beta,
            "requires_grad": self.requires_grad,
        }
        self.gradients = {"delta": 0, "alpha": 0, "beta": 0}
        self.forward_cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.where(
            x > self.delta, (x - self.delta) * self.beta, (x - self.delta) * self.alpha
        )

    def forward_pass(self, x: np.ndarray) -> None:
        self.forward_cache = {
            "alpha": np.where(x > self.delta, 0, x - self.delta),
            "beta": np.where(x > self.delta, x - self.delta, 0),
            "delta": -np.where(x > self.delta, self.beta, self.alpha),
        }

    def backward_pass(self, grad: np.ndarray) -> np.ndarray:
        precompute = self.forward_cache["delta"] * grad
        if self.requires_grad:
            self.gradients["alpha"] += self.forward_cache["alpha"] * grad
            self.gradients["beta"] += self.forward_cache["beta"] * grad
            self.gradients["delta"] += precompute

        return -precompute


class DropOut(Module):
    def __init__(self, ratio: float):
        super().__init__()
        self.ratio = ratio
        self.requires_grad = False
        self.name = "DropOut"
        self.parameters = {
            "ratio": ratio,
            "requires_grad": self.requires_grad,
        }
        self.gradients = {"ratio": 0}
        self.forward_cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.forward_cache["mask"] * x

    def forward_pass(self, x: np.ndarray) -> None:
        self.forward_cache = {
            "mask": (np.random.rand(*x.shape) >= self.ratio).astype(int),
        }

    def backward_pass(self, grad: np.ndarray) -> np.ndarray:
        return self.forward_cache["mask"] * grad


class SoftMax(Module):
    def __init__(self):
        super().__init__()
        self.requires_grad = False
        self.name = "SoftMax"
        self.parameters = {
            "requires_grad": self.requires_grad,
        }
        self.gradients = {}
        self.forward_cache = None

    def forward(self, _: np.ndarray) -> np.ndarray:
        return self.forward_cache["sigma"]

    def forward_pass(self, x: np.ndarray) -> None:
        e_x = np.exp(x)
        self.forward_cache = {
            "sigma": e_x / e_x.sum(),
        }

    def backward_pass(self, grad: np.ndarray) -> np.ndarray:
        sigma = self.forward_cache["sigma"]
        return sigma * (grad - (grad * sigma).sum(axis=-1, keepdims=True))


class Embedding(Module):
    """Needs completion."""

    def __init__(self, vocab_size: int, embed: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = embed
        self.requires_grad = True
        self.name = "Embedding"
        self.parameters = {
            "weights": np.random.randn(vocab_size, embed),
            "requires_grad": self.requires_grad,
        }
        self.gradients = {"weights": 0}
        self.forward_cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.parameters["weights"][x.squeeze()]

    def forward_pass(self, x: np.ndarray) -> None:
        self.forward_cache = {
            "alpha": np.where(x > self.delta, 0, x - self.delta),
            "beta": np.where(x > self.delta, x - self.delta, 0),
            "delta": -np.where(x > self.delta, self.beta, self.alpha),
        }

    def backward_pass(self, grad: np.ndarray) -> None:
        precompute = self.forward_cache["delta"] * grad
        if self.requires_grad:
            self.gradients["alpha"] += self.forward_cache["alpha"] * grad
            self.gradients["beta"] += self.forward_cache["beta"] * grad
            self.gradients["delta"] += precompute

        return -precompute


class ModuleSequence:
    def __init__(self, modules: list, optim: Callable = SimpleStep()):
        self.modules = modules
        self.optim = optim

    def __getitem__(self, idx: int) -> object:
        return self.modules[idx]

    def __setitem__(self, idx: int, module: object) -> None:
        self.modules[idx] = module

    def __len__(self) -> int:
        return len(self.modules)

    def __call__(self, x: Tensor) -> Tensor:
        for module in self.modules:
            x = module(x)
        return x

    def zero_grad(self) -> None:
        for module in self.modules:
            if hasattr(module, "zero_grad"):
                module.zero_grad()

    def step(self) -> None:
        for module in self.modules:
            if hasattr(module, "step"):
                module.step(self.optim)
