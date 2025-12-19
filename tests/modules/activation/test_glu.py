import numpy as np
import pytest
from deepi.modules.activation.glu import GLU

# --------------------------------------------------------------------------
# Tests for GLU activation
# --------------------------------------------------------------------------

def test_forward_exact():
    m = GLU(axis=-1)
    x = np.array([[1.0, 2.0, 3.0, 4.0]])
    y = m(x)
    a, b = np.split(x, 2, axis=-1)
    expected = a * (1.0 / (1.0 + np.exp(-b)))
    assert np.allclose(y, expected), "Forward output mismatch with exact formula"

def test_backward_exact():
    m = GLU(axis=-1)
    m.train()
    x = np.array([[1.0, 2.0, 3.0, 4.0]])
    m(x)
    dy = np.ones_like(x)
    dx = m.gradients(dy)
    a, b = np.split(x, 2, axis=-1)
    sigmoid_b = 1.0 / (1.0 + np.exp(-b))
    dy_a = np.split(dy, 2, axis=-1)[0]
    dy_b = np.split(dy, 2, axis=-1)[1]
    expected_dx_a = dy_a * sigmoid_b
    expected_dx_b = dy_b * a * sigmoid_b * (1.0 - sigmoid_b)
    expected_dx = np.concatenate([expected_dx_a, expected_dx_b], axis=-1)
    assert np.allclose(dx, expected_dx), "Backward gradients mismatch with exact formula"

def test_backward_accumulation():
    m = GLU(axis=-1)
    m.train()
    x = np.array([[1.0, 2.0, 3.0, 4.0]])
    m(x)
    dy1 = np.ones_like(x)
    dy2 = 2 * np.ones_like(x)
    m.backward(dy1)
    m.backward(dy2)
    total_dy = dy1 + dy2
    a, b = np.split(x, 2, axis=-1)
    sigmoid_b = 1.0 / (1.0 + np.exp(-b))
    total_dy_a = np.split(total_dy, 2, axis=-1)[0]
    total_dy_b = np.split(total_dy, 2, axis=-1)[1]
    expected_dx = np.concatenate([total_dy_a * sigmoid_b, total_dy_b * a * sigmoid_b * (1.0 - sigmoid_b)], axis=-1)
    assert np.allclose(m.gradients(m.dy), expected_dx), "Accumulated gradients incorrect"

def test_eval_mode_no_cache():
    m = GLU(axis=-1)
    m.eval()
    x = np.array([[1.0, 2.0, 3.0, 4.0]])
    y = m(x)
    assert m.x is None and m.y is None, "Cache should not store values in eval mode"

def test_train_mode_cache():
    m = GLU(axis=-1)
    m.train()
    x = np.array([[1.0, 2.0, 3.0, 4.0]])
    y = m(x)
    assert np.allclose(m.x, x)
    assert np.allclose(m.y, y)

def test_clear_resets():
    m = GLU(axis=-1)
    m.train()
    x = np.array([[1.0, 2.0, 3.0, 4.0]])
    m(x)
    dy = np.ones_like(x)
    m.backward(dy)
    m.clear()
    assert m.x is None
    assert m.y is None
    assert m.dy is None