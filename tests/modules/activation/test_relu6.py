import numpy as np
import pytest
from deepi.modules.activation.relu6 import ReLU6

# --------------------------------------------------------------------------
# Tests for ReLU6 activation
# --------------------------------------------------------------------------

def test_forward_exact():
    m = ReLU6()
    x = np.array([-2.0, 0.0, 3.0, 6.0, 8.0])
    y = m.forward(x)
    expected = np.clip(x, 0.0, 6.0)
    assert np.allclose(y, expected), "Forward output mismatch with exact np.clip"

def test_backward_exact():
    m = ReLU6()
    m.train()
    x = np.array([-1.0, 0.0, 3.0, 6.0, 7.0])
    y = m.forward(x)
    dy = np.ones_like(x)
    dx = m.gradients(dy)
    expected_dx = dy * ((x > 0.0) & (x < 6.0))
    assert np.allclose(dx, expected_dx), "Backward gradients mismatch with exact mask"

def test_backward_accumulation():
    m = ReLU6()
    m.train()
    x = np.array([-1.0, 2.0, 6.0])
    m.forward(x)
    dy1 = np.array([1.0, 2.0, 3.0])
    dy2 = np.array([4.0, 5.0, 6.0])
    m.backward(dy1)
    m.backward(dy2)
    total_dy = dy1 + dy2
    expected_dx = total_dy * ((x > 0.0) & (x < 6.0))
    assert np.allclose(m.gradients(m.dy), expected_dx), "Accumulated gradients incorrect"

def test_eval_mode_no_cache():
    m = ReLU6()
    m.eval()
    x = np.array([-1.0, 0.0, 3.0])
    y = m.forward(x)
    assert m.x is None and m.y is None, "Cache should not store values in eval mode"

def test_train_mode_cache():
    m = ReLU6()
    m.train()
    x = np.array([-1.0, 0.0, 3.0])
    y = m.forward(x)
    assert np.allclose(m.x, x)
    assert np.allclose(m.y, y)

def test_clear_resets():
    m = ReLU6()
    m.train()
    x = np.array([-1.0, 3.0, 6.0])
    m.forward(x)
    dy = np.array([1.0, 2.0, 3.0])
    m.backward(dy)
    m.clear()
    assert m.x is None
    assert m.y is None
    assert m.dy is None