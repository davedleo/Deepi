import numpy as np
import pytest
from deepi.modules.flow.concatenate import Concatenate

# --------------------------------------------------------------------------
# Tests for Concatenate flow layer
# --------------------------------------------------------------------------

def test_forward_exact():
    m = Concatenate(axis=1)
    x1 = np.array([[1, 2]])
    x2 = np.array([[3, 4]])
    y = m.forward((x1, x2))
    expected = np.array([[1, 2, 3, 4]])
    assert np.allclose(y, expected), "Forward concatenation mismatch"

def test_backward_exact():
    m = Concatenate(axis=1)
    m.train()
    x1 = np.array([[1, 2]])
    x2 = np.array([[3, 4]])
    m.forward((x1, x2))

    dy = np.ones((1, 4))
    dx1, dx2 = m.gradients(dy)
    expected_dx1 = np.ones_like(x1)
    expected_dx2 = np.ones_like(x2)
    assert np.allclose(dx1, expected_dx1), "Backward gradient for x1 incorrect"
    assert np.allclose(dx2, expected_dx2), "Backward gradient for x2 incorrect"

def test_backward_accumulation():
    m = Concatenate(axis=1)
    m.train()
    x1 = np.array([[1, 2]])
    x2 = np.array([[3, 4]])
    m.forward((x1, x2))

    dy1 = np.ones((1, 4))
    dy2 = np.full((1, 4), 2.0)

    m.backward(dy1)
    m.backward(dy2)

    dx1, dx2 = m.gradients(m.dy)
    total_dy = dy1 + dy2
    assert np.allclose(dx1, total_dy[:, :2]), "Accumulated dx1 incorrect"
    assert np.allclose(dx2, total_dy[:, 2:]), "Accumulated dx2 incorrect"

def test_eval_mode_no_cache():
    m = Concatenate(axis=0)
    m.eval()
    x1 = np.array([[1, 2]])
    x2 = np.array([[3, 4]])
    y = m.forward((x1, x2))
    assert m.x is None and m.y is None, "Eval mode should not cache values"

def test_train_mode_cache():
    m = Concatenate(axis=0)
    m.train()
    x1 = np.array([[1, 2]])
    x2 = np.array([[3, 4]])
    y = m.forward((x1, x2))
    xs = (x1, x2)
    for stored, original in zip(m.x, xs):
        assert np.allclose(stored, original)
    assert np.allclose(m.y, y)

def test_clear_resets():
    m = Concatenate(axis=1)
    m.train()
    x1 = np.array([[1, 2]])
    x2 = np.array([[3, 4]])
    m.forward((x1, x2))
    dy = np.ones((1, 4))
    m.backward(dy)

    m.clear()
    assert m.x is None
    assert m.y is None
    assert m.dy is None