import numpy as np
import pytest
from deepi.modules.linear import Linear

# Concrete subclass for testing purposes
class DummyLinear(Linear):
    def transform(self, x: np.ndarray) -> np.ndarray:
        # Simple linear transform: y = 2 * x + b (if bias exists)
        y = 2 * x
        if self._has_bias:
            y += self.params["b"]
        return y

    def gradients(self, dy: np.ndarray) -> np.ndarray:
        # Gradient: just scales incoming gradient by 2
        return 2 * dy

def test_linear_init_without_bias():
    m = DummyLinear(_type="dummy", _has_bias=False, _out_size=3)
    assert m.type == "module.linear.dummy"
    assert m._has_bias is False
    assert "b" not in m.params

def test_linear_init_with_bias():
    m = DummyLinear(_type="dummy", _has_bias=True, _out_size=4)
    assert m._has_bias is True
    assert "b" in m.params
    assert m.params["b"].shape == (1, 4)
    assert np.all(m.params["b"] == 0)

def test_forward_without_bias():
    m = DummyLinear("dummy", _has_bias=False, _out_size=2)
    x = np.array([[1.0, 2.0]])
    y = m.transform(x)
    expected = 2 * x
    np.testing.assert_array_equal(y, expected)

def test_forward_with_bias():
    m = DummyLinear("dummy", _has_bias=True, _out_size=2)
    x = np.array([[1.0, 2.0]])
    # set bias to known values
    m.params["b"][:] = [[0.5, -0.5]]
    y = m.transform(x)
    expected = 2 * x + np.array([[0.5, -0.5]])
    np.testing.assert_array_equal(y, expected)

def test_backward_scaling():
    m = DummyLinear("dummy", _has_bias=False, _out_size=2)
    dy = np.array([[1.0, -1.0]])
    grad = m.gradients(dy)
    expected = 2 * dy
    np.testing.assert_array_equal(grad, expected)

def test_bias_update_effect():
    m = DummyLinear("dummy", _has_bias=True, _out_size=2)
    x = np.array([[1.0, 2.0]])
    # simulate bias update
    m.params["b"] += np.array([[1.0, -1.0]])
    y = m.transform(x)
    expected = 2 * x + np.array([[1.0, -1.0]])
    np.testing.assert_array_equal(y, expected)