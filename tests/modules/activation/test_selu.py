import numpy as np
import pytest
from deepi.modules.activation.selu import SELU

# --------------------------------------------------------------------------
# Tests for SELU activation
# --------------------------------------------------------------------------

def test_forward_exact():
    m = SELU()
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    y = m(x)
    expected = np.where(x > 0.0, m.scale * x, m.scale * m.alpha * np.expm1(x))
    assert np.allclose(y, expected), "Forward output mismatch with exact formula"

def test_backward_exact():
    m = SELU()
    m.train()
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    m(x)
    dy = np.ones_like(x)
    dx = m.gradients(dy)
    expected_dx = np.where(x > 0.0, m.scale, m.scale * m.alpha * np.exp(x)) * dy
    assert np.allclose(dx, expected_dx), "Backward gradients mismatch with exact formula"

def test_backward_accumulation():
    m = SELU()
    m.train()
    x = np.array([-1.0, 0.5, 2.0])
    m(x)
    dy1 = np.array([1.0, 2.0, 3.0])
    dy2 = np.array([4.0, 5.0, 6.0])
    m.backward(dy1)
    m.backward(dy2)
    total_dy = dy1 + dy2
    expected_dx = np.where(x > 0.0, m.scale, m.scale * m.alpha * np.exp(x)) * total_dy
    assert np.allclose(m.gradients(m.dy), expected_dx), "Accumulated gradients incorrect"

def test_eval_mode_no_cache():
    m = SELU()
    m.eval()
    x = np.array([-1.0, 0.0, 1.0])
    y = m(x)
    assert m.x is None and m.y is None, "Cache should not store values in eval mode"

def test_train_mode_cache():
    m = SELU()
    m.train()
    x = np.array([-1.0, 0.0, 1.0])
    y = m(x)
    assert np.allclose(m.x, x)
    assert np.allclose(m.y, y)

def test_clear_resets():
    m = SELU()
    m.train()
    x = np.array([-1.0, 0.0, 1.0])
    m(x)
    dy = np.array([1.0, 2.0, 3.0])
    m.backward(dy)
    m.clear()
    assert m.x is None
    assert m.y is None
    assert m.dy is None