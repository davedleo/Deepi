import numpy as np
import pytest
from deepi.modules.activation.log_softmax import LogSoftmax

# --------------------------------------------------------------------------
# Tests for LogSoftmax activation
# --------------------------------------------------------------------------

def test_forward_exact():
    m = LogSoftmax(axis=-1)
    x = np.array([[1.0, 2.0, 3.0]])
    y = m.forward(x)
    # Compute expected manually
    x_max = np.max(x, axis=-1, keepdims=True)
    shifted = x - x_max
    exp_shifted = np.exp(shifted)
    exp_sum = np.sum(exp_shifted, axis=-1, keepdims=True)
    expected = shifted - np.log(exp_sum)
    assert np.allclose(y, expected), "Forward output mismatch with exact formula"

def test_backward_exact():
    m = LogSoftmax(axis=-1)
    m.train()
    x = np.array([[1.0, 2.0, 3.0]])
    m.forward(x)
    dy = np.ones_like(x)
    dx = m.gradients(dy)

    # Compute expected
    exp_shifted = np.exp(x - np.max(x, axis=-1, keepdims=True))
    softmax = exp_shifted / np.sum(exp_shifted, axis=-1, keepdims=True)
    dot = np.sum(dy, axis=-1, keepdims=True)
    expected_dx = dy - softmax * dot
    assert np.allclose(dx, expected_dx), "Backward gradients mismatch with exact formula"

def test_backward_accumulation():
    m = LogSoftmax(axis=-1)
    m.train()
    x = np.array([[1.0, 2.0, 3.0]])
    m.forward(x)
    dy1 = np.ones_like(x)
    dy2 = 2 * np.ones_like(x)
    m.backward(dy1)
    m.backward(dy2)
    total_dy = dy1 + dy2
    exp_shifted = np.exp(x - np.max(x, axis=-1, keepdims=True))
    softmax = exp_shifted / np.sum(exp_shifted, axis=-1, keepdims=True)
    dot = np.sum(total_dy, axis=-1, keepdims=True)
    expected_dx = total_dy - softmax * dot
    assert np.allclose(m.gradients(m.dy), expected_dx), "Accumulated gradients incorrect"

def test_eval_mode_no_cache():
    m = LogSoftmax(axis=-1)
    m.eval()
    x = np.array([[1.0, 2.0, 3.0]])
    y = m.forward(x)
    assert m.x is None and m.y is None, "Cache should not store values in eval mode"

def test_train_mode_cache():
    m = LogSoftmax(axis=-1)
    m.train()
    x = np.array([[1.0, 2.0, 3.0]])
    y = m.forward(x)
    assert np.allclose(m.x, x)
    assert np.allclose(m.y, y)

def test_clear_resets():
    m = LogSoftmax(axis=-1)
    m.train()
    x = np.array([[1.0, 2.0, 3.0]])
    m.forward(x)
    dy = np.ones_like(x)
    m.backward(dy)
    m.clear()
    assert m.x is None
    assert m.y is None
    assert m.dy is None

def test_jacobian_matches_gradients():
    m = LogSoftmax(axis=-1)
    m.train()
    x = np.array([[1.0, 2.0, 3.0]])
    m.forward(x)
    
    dy = np.array([[0.1, 0.2, 0.3]])  # upstream gradient

    # Compute expected gradient using exact formula (Jacobian-vector product)
    exp_shifted = np.exp(x - np.max(x, axis=-1, keepdims=True))
    softmax = exp_shifted / np.sum(exp_shifted, axis=-1, keepdims=True)
    dx_exact = dy - np.sum(dy, axis=-1, keepdims=True) * softmax

    dx_library = m.gradients(dy)

    assert np.allclose(dx_library, dx_exact, rtol=1e-10, atol=1e-12), \
        "Gradients do not match full Jacobian"