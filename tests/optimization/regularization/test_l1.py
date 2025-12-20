import numpy as np
import pytest
from deepi.optimization.regularization import L1

# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

def test_regularization_sign():
    """L1 regularization should return sign(x)"""
    r = L1(gamma=1.0)
    x = np.array([-2.0, -0.5, 0.0, 3.0])

    reg = r.regularization(x)

    expected = np.array([-1.0, -1.0, 0.0, 1.0])
    assert np.allclose(reg, expected)


def test_call_scales_by_gamma():
    """__call__ must scale sign(x) by gamma"""
    gamma = 0.1
    r = L1(gamma=gamma)
    x = np.array([-2.0, 3.0])

    y = r(x)

    expected = gamma * np.sign(x)
    assert np.allclose(y, expected)


def test_zero_input():
    """Zero input should return zero"""
    r = L1(gamma=1.0)
    x = np.zeros(5)

    y = r(x)

    assert np.allclose(y, np.zeros_like(x))


def test_input_not_modified():
    """L1 regularizer must not modify input array"""
    r = L1(gamma=1.0)
    x = np.array([-1.0, 2.0, -3.0])
    x_copy = x.copy()

    _ = r(x)

    assert np.allclose(x, x_copy)


def test_default_gamma():
    """Default gamma value should be 0.001"""
    r = L1()

    assert np.isclose(r.gamma, 0.001)


def test_type_string():
    """Regularizer type should be correctly prefixed"""
    r = L1()

    assert r._type == "regularizer.l1"