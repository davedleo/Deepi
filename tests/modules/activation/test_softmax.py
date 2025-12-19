import numpy as np
import pytest
from deepi.modules.activation.softmax import Softmax

# --------------------------------------------------------------------------
# Tests for Softmax activation
# --------------------------------------------------------------------------

def test_forward_exact():
    m = Softmax(axis=-1)
    x = np.array([[1.0, 2.0, 3.0]])
    y = m(x)
    x_max = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - x_max)
    expected = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    assert np.allclose(y, expected), "Forward output mismatch with exact formula"

def test_backward_exact():
    m = Softmax(axis=-1)
    m.train()
    x = np.array([[1.0, 2.0, 3.0]])
    m(x)
    dy = np.array([[0.1, 0.2, 0.3]])
    dx = m.gradients(dy)

    softmax = m.y
    dot = np.sum(softmax * dy, axis=-1, keepdims=True)
    expected_dx = softmax * (dy - dot)
    assert np.allclose(dx, expected_dx), "Backward gradients mismatch with exact formula"

def test_backward_accumulation():
    m = Softmax(axis=-1)
    m.train()
    x = np.array([[1.0, 2.0, 3.0]])
    m(x)
    dy1 = np.array([[1.0, 2.0, 3.0]])
    dy2 = np.array([[4.0, 5.0, 6.0]])
    m.backward(dy1)
    m.backward(dy2)
    total_dy = dy1 + dy2
    softmax = m.y
    dot = np.sum(softmax * total_dy, axis=-1, keepdims=True)
    expected_dx = softmax * (total_dy - dot)
    assert np.allclose(m.gradients(m.dy), expected_dx), "Accumulated gradients incorrect"

def test_eval_mode_no_cache():
    m = Softmax(axis=-1)
    m.eval()
    x = np.array([[1.0, 2.0, 3.0]])
    y = m(x)
    assert m.x is None and m.y is None, "Cache should not store values in eval mode"

def test_train_mode_cache():
    m = Softmax(axis=-1)
    m.train()
    x = np.array([[1.0, 2.0, 3.0]])
    y = m(x)
    assert np.allclose(m.x, x)
    assert np.allclose(m.y, y)

def test_clear_resets():
    m = Softmax(axis=-1)
    m.train()
    x = np.array([[1.0, 2.0, 3.0]])
    m(x)
    dy = np.ones_like(x)
    m.backward(dy)
    m.clear()
    assert m.x is None
    assert m.y is None
    assert m.dy is None

def test_jacobian_vector_product():
    m = Softmax(axis=-1)
    m.train()
    x = np.array([[0.5, 1.0, 1.5]])
    m(x)
    dy = np.array([[0.1, 0.2, 0.3]])

    # Full Jacobian for small inputs
    softmax = m.y
    n = x.shape[-1]
    J = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                J[i, j] = softmax[0, i] * (1 - softmax[0, j])
            else:
                J[i, j] = -softmax[0, i] * softmax[0, j]
    dx_exact = J @ dy[0]
    dx_library = m.gradients(dy)[0]

    assert np.allclose(dx_library, dx_exact, rtol=1e-10, atol=1e-12), \
        "Gradients do not match full Jacobian"