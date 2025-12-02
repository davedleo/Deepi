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
    return np.arange(15, dtype=float).reshape(5, 3) - 7.0


@pytest.fixture
def dy():
    return 0.1 * (np.arange(15, dtype=float).reshape(5, 3) - 7.0)


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
    y = activation.forward(x)
    assert isinstance(y, np.ndarray)
    assert np.all(y == x + 1.0)
    assert not activation._is_training
    assert activation.dx == 0.0

    # Training
    activation.train()
    y_with_train = activation.forward(x)
    assert isinstance(y_with_train, np.ndarray)
    assert np.all(y_with_train == y)
    assert activation._is_training
    assert np.all(activation.dx == np.ones_like(x))

    # Backward 
    dy_new = activation.backward(dy)
    assert isinstance(dy_new, np.ndarray)
    assert np.all(dy_new == dy * 1.0)