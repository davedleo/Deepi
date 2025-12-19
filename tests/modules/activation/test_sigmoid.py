import numpy as np
import pytest
from deepi.modules.activation.sigmoid import Sigmoid

# --------------------------------------------------------------------------
# Tests for Sigmoid activation
# --------------------------------------------------------------------------

def test_forward_exact():
    m = Sigmoid()
    x = np.array([-10.0, -1.0, 0.0, 1.0, 10.0])
    y = m(x)
    expected = 1 / (1 + np.exp(-x))
    # Use np.clip to avoid overflow for large negative values
    expected = np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))
    assert np.allclose(y, expected), "Forward output mismatch with exact formula"

def test_backward_exact():
    m = Sigmoid()
    m.train()
    x = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    y = m(x)
    dy = np.ones_like(x)
    dx = m.gradients(dy)
    expected_dx = dy * y * (1 - y)
    assert np.allclose(dx, expected_dx), "Backward gradients mismatch with exact formula"

def test_backward_accumulation():
    m = Sigmoid()
    m.train()
    x = np.array([-1.0, 0.0, 1.0])
    m(x)
    dy1 = np.array([1.0, 2.0, 3.0])
    dy2 = np.array([4.0, 5.0, 6.0])
    m.backward(dy1)
    m.backward(dy2)
    total_dy = dy1 + dy2
    expected_dx = total_dy * m.y * (1 - m.y)
    assert np.allclose(m.gradients(m.dy), expected_dx), "Accumulated gradients incorrect"

def test_eval_mode_no_cache():
    m = Sigmoid()
    m.eval()
    x = np.array([-1.0, 0.0, 1.0])
    y = m(x)
    assert m.x is None and m.y is None, "Cache should not store values in eval mode"

def test_train_mode_cache():
    m = Sigmoid()
    m.train()
    x = np.array([-1.0, 0.0, 1.0])
    y = m(x)
    assert np.allclose(m.x, x)
    assert np.allclose(m.y, y)

def test_clear_resets():
    m = Sigmoid()
    m.train()
    x = np.array([-1.0, 0.0, 1.0])
    m(x)
    dy = np.array([1.0, 2.0, 3.0])
    m.backward(dy)
    m.clear()
    assert m.x is None
    assert m.y is None
    assert m.dy is None