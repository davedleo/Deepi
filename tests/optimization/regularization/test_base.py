import numpy as np
import pytest
from deepi.optimization.regularization import Regularizer

# --------------------------------------------------------------------------
# Dummy regularizer for testing
# --------------------------------------------------------------------------

class DummyRegularizer(Regularizer):
    def __init__(self, gamma: float):
        super().__init__(gamma=gamma, _type="dummy")

    def regularization(self, x: np.ndarray) -> np.ndarray:
        # simple element-wise square
        return x ** 2


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

def test_regularization_function():
    """regularization should return the raw (unscaled) regularization"""
    r = DummyRegularizer(gamma=1.0)
    x = np.array([1., -2., 3.])

    reg = r.regularization(x)

    assert np.allclose(reg, np.array([1., 4., 9.]))


def test_call_scales_by_gamma():
    """__call__ must scale regularization by gamma"""
    gamma = 0.5
    r = DummyRegularizer(gamma=gamma)
    x = np.array([2., -4.])

    y = r(x)

    expected = gamma * (x ** 2)
    assert np.allclose(y, expected)


def test_gamma_zero_gives_zero_regularization():
    """Gamma = 0 should always return zeros"""
    r = DummyRegularizer(gamma=0.0)
    x = np.array([1., 2., 3.])

    y = r(x)

    assert np.allclose(y, np.zeros_like(x))


def test_input_not_modified():
    """Regularizer must not modify input array"""
    r = DummyRegularizer(gamma=1.0)
    x = np.array([1., 2., 3.])
    x_copy = x.copy()

    _ = r(x)

    assert np.allclose(x, x_copy)


def test_type_string():
    """Regularizer should prefix type correctly"""
    r = DummyRegularizer(gamma=1.0)

    assert r._type == "regularizer.dummy"


def test_abstract_base_class_cannot_be_instantiated():
    """Regularizer base class must remain abstract"""
    with pytest.raises(TypeError):
        Regularizer(gamma=1.0, _type="base")