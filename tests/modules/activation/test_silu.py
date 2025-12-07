import numpy as np
import pytest
from deepi.modules.activation.silu import SiLU

# --------------------------------------------------------------------------
# Tests for SiLU (Sigmoid Linear Unit) activation
# --------------------------------------------------------------------------

def test_forward_exact():
    m = SiLU()
    x = np.array([-10.0, -1.0, 0.0, 1.0, 10.0])
    y = m.forward(x)
    # Compute expected using numerically stable sigmoid
    sigmoid_x = np.empty_like(x, dtype=np.float64)
    mask = x >= 0
    sigmoid_x[mask] = 1.0 / (1.0 + np.exp(-x[mask]))
    exp_x = np.exp(x[~mask])
    sigmoid_x[~mask] = exp_x / (1.0 + exp_x)
    expected = x * sigmoid_x
    assert np.allclose(y, expected), "Forward output mismatch with exact SiLU formula"

def test_backward_exact():
    m = SiLU()
    m.train()
    x = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    m.forward(x)
    dy = np.ones_like(x)
    sigmoid_x = np.empty_like(x, dtype=np.float64)
    mask = x >= 0
    sigmoid_x[mask] = 1.0 / (1.0 + np.exp(-x[mask]))
    exp_x = np.exp(x[~mask])
    sigmoid_x[~mask] = exp_x / (1.0 + exp_x)
    expected_dy_silu = sigmoid_x * (1.0 + x * (1.0 - sigmoid_x)) * dy
    dy_silu = m.gradients(dy)
    assert np.allclose(dy_silu, expected_dy_silu), "Backward gradients mismatch with exact formula"

def test_backward_accumulation():
    m = SiLU()
    m.train()
    x = np.array([-1.0, 0.0, 1.0])
    m.forward(x)
    dy1 = np.array([1.0, 2.0, 3.0])
    dy2 = np.array([4.0, 5.0, 6.0])
    m.backward(dy1)
    m.backward(dy2)
    total_dy = dy1 + dy2
    sigmoid_x = np.empty_like(x, dtype=np.float64)
    mask = x >= 0
    sigmoid_x[mask] = 1.0 / (1.0 + np.exp(-x[mask]))
    exp_x = np.exp(x[~mask])
    sigmoid_x[~mask] = exp_x / (1.0 + exp_x)
    expected_dy_silu = sigmoid_x * (1.0 + x * (1.0 - sigmoid_x)) * total_dy
    assert np.allclose(m.gradients(m.dy), expected_dy_silu), "Accumulated gradients incorrect"

def test_eval_mode_no_cache():
    m = SiLU()
    m.eval()
    x = np.array([-1.0, 0.0, 1.0])
    y = m.forward(x)
    assert m.x is None and m.y is None, "Cache should not store values in eval mode"

def test_train_mode_cache():
    m = SiLU()
    m.train()
    x = np.array([-1.0, 0.0, 1.0])
    y = m.forward(x)
    assert np.allclose(m.x, x)
    assert np.allclose(m.y, y)

def test_clear_resets():
    m = SiLU()
    m.train()
    x = np.array([-1.0, 0.0, 1.0])
    m.forward(x)
    dy = np.array([1.0, 2.0, 3.0])
    m.backward(dy)
    m.clear()
    assert m.x is None
    assert m.y is None
    assert m.dy is None