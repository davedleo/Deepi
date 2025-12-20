import numpy as np
import pytest
from deepi.optimization.regularization import ElasticNet

# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

def test_regularization_combination():
    """ElasticNet regularization should combine L1 and L2 terms correctly"""
    alpha = 0.3
    r = ElasticNet(alpha=alpha, gamma=1.0)
    x = np.array([-2.0, -1.0, 0.0, 2.0])

    reg = r.regularization(x)

    l1 = np.sign(x)
    l2 = 2.0 * x
    expected = alpha * l1 + alpha * l2

    assert np.allclose(reg, expected)


def test_call_scales_by_gamma():
    """__call__ must scale elastic net regularization by gamma"""
    alpha = 0.5
    gamma = 0.1
    r = ElasticNet(alpha=alpha, gamma=gamma)
    x = np.array([-1.0, 2.0])

    y = r(x)

    expected = gamma * (alpha * np.sign(x) + alpha * 2.0 * x)
    assert np.allclose(y, expected)


def test_zero_input():
    """Zero input should return zero"""
    r = ElasticNet(alpha=0.7, gamma=1.0)
    x = np.zeros(5)

    y = r(x)

    assert np.allclose(y, np.zeros_like(x))


def test_input_not_modified():
    """ElasticNet regularizer must not modify input array"""
    r = ElasticNet(alpha=0.5, gamma=1.0)
    x = np.array([-1.0, 2.0, -3.0])
    x_copy = x.copy()

    _ = r(x)

    assert np.allclose(x, x_copy)


def test_default_parameters():
    """Default alpha and gamma values should be correct"""
    r = ElasticNet()

    assert np.isclose(r.alpha, 0.5)
    assert np.isclose(r.gamma, 0.001)


def test_invalid_alpha_raises():
    """Alpha outside [0, 1] must raise ValueError"""
    with pytest.raises(ValueError):
        ElasticNet(alpha=-0.1)

    with pytest.raises(ValueError):
        ElasticNet(alpha=1.1)


def test_type_string():
    """Regularizer type should be correctly prefixed"""
    r = ElasticNet()

    assert r._type == "regularizer.elastic_net"