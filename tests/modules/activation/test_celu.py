import numpy as np
import pytest
from deepi.modules.activation.celu import CELU

# --------------------------------------------------------------------------
# Tests for CELU activation
# --------------------------------------------------------------------------

def test_forward_exact():
    m = CELU(alpha=1.0)
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    y = m(x)
    expected = np.where(x > 0.0, x, 1.0 * np.expm1(x / 1.0))
    assert np.allclose(y, expected), "Forward output mismatch with exact formula"

def test_backward_exact():
    m = CELU(alpha=1.0)
    m.train()
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    m(x)
    dy = np.ones_like(x)
    dx = m.gradients(dy)
    expected_dx = np.where(x > 0.0, 1.0, np.exp(x / 1.0))
    assert np.allclose(dx, expected_dx), "Backward gradients mismatch with exact formula"

def test_backward_accumulation():
    m = CELU(alpha=1.0)
    m.train()
    x = np.array([-1.0, 0.5, 2.0])
    m(x)
    dy1 = np.array([1.0, 2.0, 3.0])
    dy2 = np.array([4.0, 5.0, 6.0])
    m.backward(dy1)
    m.backward(dy2)
    total_dy = dy1 + dy2
    expected_dx = total_dy * np.where(x > 0.0, 1.0, np.exp(x / 1.0))
    assert np.allclose(m.gradients(m.dy), expected_dx), "Accumulated gradients incorrect"

def test_eval_mode_no_cache():
    m = CELU(alpha=1.0)
    m.eval()
    x = np.array([-1.0, 0.0, 1.0])
    y = m(x)
    assert m.x is None and m.y is None, "Cache should not store values in eval mode"

def test_train_mode_cache():
    m = CELU(alpha=1.0)
    m.train()
    x = np.array([-1.0, 0.0, 1.0])
    y = m(x)
    assert np.allclose(m.x, x)
    assert np.allclose(m.y, y)

def test_clear_resets():
    m = CELU(alpha=1.0)
    m.train()
    x = np.array([-1.0, 0.0, 1.0])
    m(x)
    dy = np.array([1.0, 2.0, 3.0])
    m.backward(dy)
    m.clear()
    assert m.x is None
    assert m.y is None
    assert m.dy is None