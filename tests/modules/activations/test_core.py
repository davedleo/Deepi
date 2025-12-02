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
    Softmax
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


def test_gelu(x, dy):
    # Test GELU activation with approximate=True and approximate=False

    for approximate in [True, False]:
        activation = GELU(approximate=approximate)

        # Initialization
        assert activation.type == "module.activation.gelu"
        assert not activation._is_training
        assert activation.dx == 0.0

        # No training
        out = activation.forward(x)
        assert isinstance(out, np.ndarray)
        assert out.shape == x.shape
        assert not activation._is_training
        assert activation.dx == 0.0

        # Training
        activation.train()
        out_train = activation.forward(x)
        assert isinstance(out_train, np.ndarray)
        assert out_train.shape == x.shape
        assert activation._is_training
        assert isinstance(activation.dx, np.ndarray)
        assert activation.dx.shape == x.shape

        # Backward
        dx_new = activation.backward(dy)
        assert isinstance(dx_new, np.ndarray)
        assert dx_new.shape == x.shape


# leaky_relu
def test_leaky_relu(x, dy):
    activation = LeakyReLU(alpha=0.01)

    # Initialization
    assert activation.type == "module.activation.leaky_relu"
    assert not activation._is_training
    assert activation.dx == 0.0

    # No training
    out = activation.forward(x)
    expected = np.where(x > 0, x, 0.01 * x)
    assert isinstance(out, np.ndarray)
    assert np.allclose(out, expected)
    assert not activation._is_training
    assert activation.dx == 0.0

    # Training
    activation.train()
    out_train = activation.forward(x)
    assert isinstance(out_train, np.ndarray)
    assert np.allclose(out_train, expected)
    assert activation._is_training
    assert np.allclose(
        activation.dx,
        np.where(x > 0, 1.0, 0.01)
    )

    # Backward
    dx_new = activation.backward(dy)
    assert isinstance(dx_new, np.ndarray)
    assert np.allclose(dx_new, dy * np.where(x > 0, 1.0, 0.01))


# ReLU
def test_relu(x, dy):
    activation = ReLU()

    # Initialization
    assert activation.type == "module.activation.relu"
    assert not activation._is_training
    assert activation.dx == 0.0

    # No training
    out = activation.forward(x)
    expected = np.maximum(x, 0)
    assert isinstance(out, np.ndarray)
    assert np.allclose(out, expected)
    assert not activation._is_training
    assert activation.dx == 0.0

    # Training
    activation.train()
    out_train = activation.forward(x)
    assert isinstance(out_train, np.ndarray)
    assert np.allclose(out_train, expected)
    assert activation._is_training
    assert np.allclose(
        activation.dx,
        (x > 0).astype(float)
    )

    # Backward
    dx_new = activation.backward(dy)
    assert isinstance(dx_new, np.ndarray)
    assert np.allclose(dx_new, dy * (x > 0).astype(float))


# ReLU6
def test_relu6(x, dy):
    activation = ReLU6()

    # Initialization
    assert activation.type == "module.activation.relu6"
    assert not activation._is_training
    assert activation.dx == 0.0

    # No training
    out = activation.forward(x)
    expected = np.clip(x, 0, 6)
    assert isinstance(out, np.ndarray)
    assert np.allclose(out, expected)
    assert not activation._is_training
    assert activation.dx == 0.0

    # Training
    activation.train()
    out_train = activation.forward(x)
    assert isinstance(out_train, np.ndarray)
    assert np.allclose(out_train, expected)
    assert activation._is_training
    mask = ((x > 0) & (x < 6)).astype(float)
    assert np.allclose(activation.dx, mask)

    # Backward
    dx_new = activation.backward(dy)
    assert isinstance(dx_new, np.ndarray)
    assert np.allclose(dx_new, dy * mask)


# SELU
def test_selu(x, dy):
    activation = SELU()
    # Standard SELU constants
    alpha = 1.6732632423543772
    scale = 1.0507009873554805

    # Initialization
    assert activation.type == "module.activation.selu"
    assert not activation._is_training
    assert activation.dx == 0.0

    # No training
    out = activation.forward(x)
    expected = scale * np.where(x > 0, x, alpha * (np.exp(x) - 1))
    assert isinstance(out, np.ndarray)
    assert np.allclose(out, expected)
    assert not activation._is_training
    assert activation.dx == 0.0

    # Training
    activation.train()
    out_train = activation.forward(x)
    assert isinstance(out_train, np.ndarray)
    assert np.allclose(out_train, expected)
    assert activation._is_training
    dx_expected = scale * np.where(x > 0, 1.0, alpha * np.exp(x))
    assert np.allclose(activation.dx, dx_expected)

    # Backward
    dx_new = activation.backward(dy)
    assert isinstance(dx_new, np.ndarray)
    assert np.allclose(dx_new, dy * dx_expected)


# Sigmoid
def test_sigmoid(x, dy):
    activation = Sigmoid()

    # Initialization
    assert activation.type == "module.activation.sigmoid"
    assert not activation._is_training
    assert activation.dx == 0.0

    # No training
    out = activation.forward(x)
    sig = 1.0 / (1.0 + np.exp(-x))
    expected = sig
    assert isinstance(out, np.ndarray)
    assert np.allclose(out, expected)
    assert not activation._is_training
    assert activation.dx == 0.0

    # Training
    activation.train()
    out_train = activation.forward(x)
    sig = 1.0 / (1.0 + np.exp(-x))
    assert isinstance(out_train, np.ndarray)
    assert np.allclose(out_train, sig)
    assert activation._is_training
    dx_expected = sig * (1.0 - sig)
    assert np.allclose(activation.dx, dx_expected)

    # Backward
    dx_new = activation.backward(dy)
    assert isinstance(dx_new, np.ndarray)
    assert np.allclose(dx_new, dy * dx_expected)


# SiLU
def test_silu(x, dy):
    activation = SiLU()

    # Initialization
    assert activation.type == "module.activation.silu"
    assert not activation._is_training
    assert activation.dx == 0.0

    # No training
    out = activation.forward(x)
    sig = 1.0 / (1.0 + np.exp(-x))
    expected = x * sig
    assert isinstance(out, np.ndarray)
    assert np.allclose(out, expected)
    assert not activation._is_training
    assert activation.dx == 0.0

    # Training
    activation.train()
    out_train = activation.forward(x)
    sig = 1.0 / (1.0 + np.exp(-x))
    dx_expected = sig * (1 + x * (1 - sig))
    assert isinstance(out_train, np.ndarray)
    assert np.allclose(out_train, x * sig)
    assert activation._is_training
    assert np.allclose(activation.dx, dx_expected)

    # Backward
    dx_new = activation.backward(dy)
    assert isinstance(dx_new, np.ndarray)
    assert np.allclose(dx_new, dy * dx_expected)


# Swish
def test_swish(x, dy):
    activation = Swish()

    # Initialization
    assert activation.type == "module.activation.swish"
    assert not activation._is_training
    assert activation.dx == 0.0

    # No training
    out = activation.forward(x)
    sig = 1.0 / (1.0 + np.exp(-x))
    expected = x * sig
    assert isinstance(out, np.ndarray)
    assert np.allclose(out, expected)
    assert not activation._is_training
    assert activation.dx == 0.0

    # Training
    activation.train()
    out_train = activation.forward(x)
    sig = 1.0 / (1.0 + np.exp(-x))
    dx_expected = sig + x * sig * (1 - sig)
    assert isinstance(out_train, np.ndarray)
    assert np.allclose(out_train, x * sig)
    assert activation._is_training
    assert np.allclose(activation.dx, dx_expected)

    # Backward
    dx_new = activation.backward(dy)
    assert isinstance(dx_new, np.ndarray)
    assert np.allclose(dx_new, dy * dx_expected)


# Tanh
def test_tanh(x, dy):
    activation = Tanh()

    # Initialization
    assert activation.type == "module.activation.tanh"
    assert not activation._is_training
    assert activation.dx == 0.0

    # No training
    out = activation.forward(x)
    t = np.tanh(x)
    expected = t
    assert isinstance(out, np.ndarray)
    assert np.allclose(out, expected)
    assert not activation._is_training
    assert activation.dx == 0.0

    # Training
    activation.train()
    out_train = activation.forward(x)
    t = np.tanh(x)
    dx_expected = 1.0 - t ** 2
    assert isinstance(out_train, np.ndarray)
    assert np.allclose(out_train, t)
    assert activation._is_training
    assert np.allclose(activation.dx, dx_expected)

    # Backward
    dx_new = activation.backward(dy)
    assert isinstance(dx_new, np.ndarray)
    assert np.allclose(dx_new, dy * dx_expected)


# Softmax
def test_softmax(x, dy):
    # Test Softmax activation along the last axis
    activation = Softmax(axis=1)

    # Initialization
    assert activation.type == "module.activation.softmax"
    assert not activation._is_training
    assert activation.dx == 0.0

    # No training
    out = activation.forward(x)
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    expected = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    assert isinstance(out, np.ndarray)
    assert out.shape == x.shape
    assert np.allclose(out, expected)
    assert not activation._is_training
    assert activation.dx == 0.0

    # Training
    activation.train()
    out_train = activation.forward(x)
    assert isinstance(out_train, np.ndarray)
    assert out_train.shape == x.shape
    assert activation._is_training
    dx_expected = expected
    assert np.allclose(activation.dx, dx_expected)

    # Backward
    dx_new = activation.backward(dy)
    assert isinstance(dx_new, np.ndarray)
    # Manually compute full softmax Jacobian for verification
    batch_size, _ = x.shape
    expected_dx = np.zeros_like(x)
    for b in range(batch_size):
        s = expected[b].reshape(-1, 1)  # column vector
        J = np.diagflat(s) - s @ s.T    # full Jacobian
        expected_dx[b] = J @ dy[b]
    assert np.allclose(dx_new, expected_dx)