import numpy as np
import pytest
from deepi.optimization.regularization import L2

# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

def test_regularization_linear():
    """L2 regularization should return 2 * x"""
    r = L2(gamma=1.0)
    x = np.array([-2.0, -0.5, 0.0, 3.0])

    reg = r.regularization(x)

    expected = 2.0 * x
    assert np.allclose(reg, expected)


def test_call_scales_by_gamma():
    """__call__ must scale 2 * x by gamma"""
    gamma = 0.1
    r = L2(gamma=gamma)
    x = np.array([-2.0, 3.0])

    y = r(x)

    expected = gamma * 2.0 * x
    assert np.allclose(y, expected)


def test_zero_input():
    """Zero input should return zero"""
    r = L2(gamma=1.0)
    x = np.zeros(5)

    y = r(x)

    assert np.allclose(y, np.zeros_like(x))


def test_input_not_modified():
    """L2 regularizer must not modify input array"""
    r = L2(gamma=1.0)
    x = np.array([-1.0, 2.0, -3.0])
    x_copy = x.copy()

    _ = r(x)

    assert np.allclose(x, x_copy)


def test_default_gamma():
    """Default gamma value should be 0.001"""
    r = L2()

    assert np.isclose(r.gamma, 0.001)


def test_type_string():
    """Regularizer type should be correctly prefixed"""
    r = L2()

    assert r._type == "regularizer.l2"