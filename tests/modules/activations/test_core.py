import pytest
import numpy as np
from deepi.modules.activations import (
    Activation,
    CELU,
    ELU,
    GELU,
    GLU,
    LeakyReLU,
    ReLU,
    ReLU6,
    SELU,
    Sigmoid,
    SiLU,
    Swish,
    Tanh,
)


@pytest.fixture
def x():
    return np.arange(20, dtype=float).reshape(5, 4) - 10.0


@pytest.fixture
def dy():
    return 0.1 * (np.arange(20, dtype=float).reshape(5, 4) - 10.0)


class DummyActivation(Activation):

    def __init__(self):
        super().__init__("dummy")

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self._is_training:
            self.dx = np.ones_like(x)
        return x + 1

    def backward(self, dy: np.ndarray) -> np.ndarray:
        return dy * self.dx


def test_activation(x, dy):

    activation = DummyActivation()

    # Initialization

    assert activation.type == "module.activation.dummy"
    assert not activation._is_training
    assert activation.dx == 0.0

    # No training

    out = activation.forward(x)
    assert isinstance(out, np.ndarray)
    assert np.all(out == x + 1.0)
    assert not activation._is_training
    assert activation.dx == 0.0

    # Training

    activation.train()
    out_with_train = activation.forward(x)
    assert isinstance(out_with_train, np.ndarray)
    assert np.all(out_with_train == out)
    assert activation._is_training
    assert np.all(activation.dx == np.ones_like(x))

    # Backward

    dx_new = activation.backward(dy)
    assert isinstance(dx_new, np.ndarray)
    assert np.all(dx_new == dy * 1.0)


def test_celu(x, dy):

    activation = CELU(alpha=1.0)

    # Initialization

    assert activation.type == "module.activation.celu"
    assert not activation._is_training
    assert activation.dx == 0.0

    # No training

    out = activation.forward(x)
    expected = np.where(x > 0.0, x, np.expm1(x / 1.0))
    assert isinstance(out, np.ndarray)
    assert np.all(out == expected)
    assert not activation._is_training
    assert activation.dx == 0.0

    # Training

    activation.train()
    out_with_train = activation.forward(x)
    expected_train = np.where(x > 0.0, x, np.expm1(x / 1.0))
    assert isinstance(out_with_train, np.ndarray)
    assert np.all(out_with_train == expected_train)
    assert activation._is_training
    assert np.all(activation.dx == np.where(x > 0.0, 1.0, np.exp(x / 1.0)))

    # Backward

    dx_new = activation.backward(dy)
    assert isinstance(dx_new, np.ndarray)
    assert np.all(dx_new == dy * np.where(x > 0.0, 1.0, np.exp(x / 1.0)))


def test_elu(x, dy):

    activation = ELU(alpha=1.0)

    # Initialization

    assert activation.type == "module.activation.elu"
    assert not activation._is_training
    assert activation.dx == 0.0

    # No training

    out = activation.forward(x)
    expected = np.where(x > 0.0, x, 1.0 * (np.exp(x) - 1))
    assert isinstance(out, np.ndarray)
    assert np.allclose(out, expected)
    assert not activation._is_training
    assert activation.dx == 0.0

    # Training

    activation.train()
    out_with_train = activation.forward(x)
    expected_train = np.where(x > 0.0, x, 1.0 * (np.exp(x) - 1))
    assert isinstance(out_with_train, np.ndarray)
    assert np.allclose(out_with_train, expected_train)
    assert activation._is_training
    assert np.allclose(
        activation.dx,
        np.where(x > 0.0, 1.0, 1.0 * np.exp(x))
    )

    # Backward

    dx_new = activation.backward(dy)
    assert isinstance(dx_new, np.ndarray)
    assert np.allclose(dx_new, dy * np.where(x > 0.0, 1.0, 1.0 * np.exp(x)))


def test_glu(x, dy):
    # Ensure x has even number of columns for GLU
    x_even = np.arange(20, dtype=float).reshape(5, 4) - 7.0
    dy_even = 0.1 * (np.arange(20, dtype=float).reshape(5, 4) - 7.0)

    activation = GLU(axis=1)

    # Initialization

    assert activation.type == "module.activation.glu"
    assert not activation._is_training
    assert activation.dx == 0.0

    # No training

    out = activation.forward(x_even)
    half = x_even.shape[1] // 2
    x_a = x_even[:, :half]
    x_b = x_even[:, half:]
    expected = x_a * (1 / (1 + np.exp(-x_b)))
    assert isinstance(out, np.ndarray)
    assert np.allclose(out, expected)
    assert not activation._is_training
    assert activation.dx == 0.0

    # Training

    activation.train()
    out_with_train = activation.forward(x_even)
    expected_train = x_a * (1 / (1 + np.exp(-x_b)))
    assert isinstance(out_with_train, np.ndarray)
    assert np.allclose(out_with_train, expected_train)
    assert activation._is_training

    sigmoid = 1 / (1 + np.exp(-x_b))
    grad_a = sigmoid
    grad_b = x_a * sigmoid * (1 - sigmoid)
    expected_dx = np.hstack([grad_a, grad_b])
    assert np.allclose(activation.dx, expected_dx)

    # Backward

    dx_new = activation.backward(dy_even)
    assert isinstance(dx_new, np.ndarray)
    assert np.allclose(dx_new, dy_even * expected_dx)