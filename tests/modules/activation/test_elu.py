import numpy as np
import pytest
from deepi.modules.activation.elu import ELU

# --------------------------------------------------------------------------
# Tests for ELU activation
# --------------------------------------------------------------------------

def test_forward_exact():
    alpha = 1.0
    m = ELU(alpha=alpha)
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    y = m(x)
    expected = np.where(x > 0.0, x, alpha * (np.exp(x) - 1))
    assert np.allclose(y, expected), "Forward output mismatch with exact ELU formula"

def test_backward_exact():
    alpha = 1.0
    m = ELU(alpha=alpha)
    m.train()
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    m(x)
    dy = np.ones_like(x)
    dy_elu = np.empty_like(x, dtype=np.float64)
    mask = x > 0.0
    dy_elu[mask] = 1.0
    dy_elu[~mask] = alpha * np.exp(x[~mask])
    expected = dy_elu * dy
    dx = m.gradients(dy)
    assert np.allclose(dx, expected), "Backward gradients mismatch with exact ELU formula"

def test_backward_accumulation():
    alpha = 1.0
    m = ELU(alpha=alpha)
    m.train()
    x = np.array([-1.0, 0.0, 1.0])
    m(x)
    dy1 = np.array([1.0, 2.0, 3.0])
    dy2 = np.array([4.0, 5.0, 6.0])
    m.backward(dy1)
    m.backward(dy2)
    total_dy = dy1 + dy2
    dy_elu = np.empty_like(x, dtype=np.float64)
    mask = x > 0.0
    dy_elu[mask] = 1.0
    dy_elu[~mask] = alpha * np.exp(x[~mask])
    expected = dy_elu * total_dy
    assert np.allclose(m.gradients(m.dy), expected), "Accumulated gradients incorrect"

def test_eval_mode_no_cache():
    m = ELU()
    m.eval()
    x = np.array([-1.0, 0.0, 1.0])
    y = m(x)
    assert m.x is None and m.y is None, "Cache should not store values in eval mode"

def test_train_mode_cache():
    m = ELU()
    m.train()
    x = np.array([-1.0, 0.0, 1.0])
    y = m(x)
    assert np.allclose(m.x, x)
    assert np.allclose(m.y, y)

def test_clear_resets():
    m = ELU()
    m.train()
    x = np.array([-1.0, 0.0, 1.0])
    m(x)
    dy = np.array([1.0, 2.0, 3.0])
    m.backward(dy)
    m.clear()
    assert m.x is None
    assert m.y is None
    assert m.dy is None