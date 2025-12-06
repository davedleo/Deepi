import numpy as np
import pytest
from deepi.modules.activation import Activation

# --------------------------------------------------------------------------
# Dummy activation for testing
# --------------------------------------------------------------------------

class DummyActivation(Activation):
    def __init__(self):
        super().__init__("dummy")

    def transform(self, x: np.ndarray) -> np.ndarray:
        return 2 * x

    def gradients(self, dy: np.ndarray) -> np.ndarray:
        return 2 * dy

# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

def test_forward_without_training():
    m = DummyActivation()
    x = np.array([1., 2., 3.])
    y = m.forward(x)
    assert np.allclose(y, [2., 4., 6.])
    assert m.x is None
    assert m.y is None

def test_forward_with_training_caching():
    m = DummyActivation()
    m.train()
    x = np.array([1., 3.])
    y = m.forward(x)
    assert np.allclose(y, [2., 6.])
    assert np.allclose(m.x, x)
    assert np.allclose(m.y, y)

def test_backward_accumulation():
    m = DummyActivation()
    m.train()
    dy1 = np.array([1., 2., 3.])
    dy2 = np.array([2., 3., 4.])
    dy1_c = dy1.copy()
    dy2_c = dy2.copy()
    m.backward(dy1)
    m.backward(dy2)
    assert np.allclose(dy1, dy1_c)
    assert np.allclose(dy2, dy2_c)
    assert np.allclose(m.dy, dy1 + dy2)
    dx = m.gradients(m.dy)
    assert np.allclose(dx, 2 * m.dy)

def test_clear_resets():
    m = DummyActivation()
    m.train()
    x = np.array([1., 2.])
    m.forward(x)
    m.backward(np.array([3., 4.]))
    m.clear()
    assert m.x is None
    assert m.y is None
    assert m.dy is None