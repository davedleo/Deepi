import numpy as np
import pytest
from deepi.modules.flow import Add

# --------------------------------------------------------------------------
# Tests for Add flow layer (element-wise sum / residual connection)
# --------------------------------------------------------------------------

def test_forward_exact():
    m = Add()
    x1 = np.array([[1, 2]])
    x2 = np.array([[3, 4]])
    y = m((x1, x2))
    expected = np.array([[4, 6]])
    assert np.allclose(y, expected), "Forward addition mismatch"

def test_backward_exact():
    m = Add()
    m.train()
    x1 = np.array([[1, 2]])
    x2 = np.array([[3, 4]])
    m((x1, x2))

    dy = np.ones((1, 2))
    dx1, dx2 = m.gradients(dy)
    expected = np.ones_like(x1)
    assert np.allclose(dx1, expected), "Backward gradient for x1 incorrect"
    assert np.allclose(dx2, expected), "Backward gradient for x2 incorrect"

def test_backward_accumulation():
    m = Add()
    m.train()
    x1 = np.array([[1, 2]])
    x2 = np.array([[3, 4]])
    m((x1, x2))

    dy1 = np.ones((1, 2))
    dy2 = np.full((1, 2), 2.0)
    m.backward(dy1)
    m.backward(dy2)

    total_dy = dy1 + dy2
    dx1, dx2 = m.gradients(m.dy)
    assert np.allclose(dx1, total_dy), "Accumulated dx1 incorrect"
    assert np.allclose(dx2, total_dy), "Accumulated dx2 incorrect"

def test_eval_mode_no_cache():
    m = Add()
    m.eval()
    x1 = np.array([[1, 2]])
    x2 = np.array([[3, 4]])
    y = m((x1, x2))
    assert m.x is None and m.y is None, "Eval mode should not cache values"

def test_train_mode_cache():
    m = Add()
    m.train()
    x1 = np.array([[1, 2]])
    x2 = np.array([[3, 4]])
    y = m((x1, x2))

    xs = (x1, x2)
    for stored, original in zip(m.x, xs):
        assert np.allclose(stored, original)
    assert np.allclose(m.y, y)

def test_clear_resets():
    m = Add()
    m.train()
    x1 = np.array([[1, 2]])
    x2 = np.array([[3, 4]])
    m((x1, x2))
    dy = np.ones((1, 2))
    m.backward(dy)

    m.clear()
    assert m.x is None
    assert m.y is None
    assert m.dy is None