import numpy as np
import pytest
from deepi.modules.activation.swish import Swish

# --------------------------------------------------------------------------
# Tests for Swish activation
# --------------------------------------------------------------------------

def test_forward_exact():
    beta = 1.0
    m = Swish(beta=beta)
    x = np.array([-10.0, -1.0, 0.0, 1.0, 10.0])
    y = m.forward(x)
    z = beta * x
    sigmoid_z = np.empty_like(z)
    mask = z >= 0
    sigmoid_z[mask] = 1.0 / (1.0 + np.exp(-z[mask]))
    exp_z = np.exp(z[~mask])
    sigmoid_z[~mask] = exp_z / (1.0 + exp_z)
    expected = x * sigmoid_z
    assert np.allclose(y, expected), "Forward output mismatch with exact Swish formula"

def test_backward_exact():
    beta = 1.0
    m = Swish(beta=beta)
    m.train()
    x = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    m.forward(x)
    dy = np.ones_like(x)
    z = beta * x
    sigmoid_z = np.empty_like(z)
    mask = z >= 0
    sigmoid_z[mask] = 1.0 / (1.0 + np.exp(-z[mask]))
    exp_z = np.exp(z[~mask])
    sigmoid_z[~mask] = exp_z / (1.0 + exp_z)
    dy_swish_expected = sigmoid_z + beta * x * sigmoid_z * (1 - sigmoid_z)
    dy_swish_expected *= dy
    dy_swish = m.gradients(dy)
    assert np.allclose(dy_swish, dy_swish_expected), "Backward gradients mismatch with exact formula"

def test_backward_accumulation():
    beta = 1.0
    m = Swish(beta=beta)
    m.train()
    x = np.array([-1.0, 0.0, 1.0])
    m.forward(x)
    dy1 = np.array([1.0, 2.0, 3.0])
    dy2 = np.array([4.0, 5.0, 6.0])
    m.backward(dy1)
    m.backward(dy2)
    total_dy = dy1 + dy2
    z = beta * x
    sigmoid_z = np.empty_like(z)
    mask = z >= 0
    sigmoid_z[mask] = 1.0 / (1.0 + np.exp(-z[mask]))
    exp_z = np.exp(z[~mask])
    sigmoid_z[~mask] = exp_z / (1.0 + exp_z)
    dx_expected = (sigmoid_z + beta * x * sigmoid_z * (1 - sigmoid_z)) * total_dy
    assert np.allclose(m.gradients(m.dy), dx_expected), "Accumulated gradients incorrect"

def test_eval_mode_no_cache():
    m = Swish()
    m.eval()
    x = np.array([-1.0, 0.0, 1.0])
    y = m.forward(x)
    assert m.x is None and m.y is None, "Cache should not store values in eval mode"

def test_train_mode_cache():
    m = Swish()
    m.train()
    x = np.array([-1.0, 0.0, 1.0])
    y = m.forward(x)
    assert np.allclose(m.x, x)
    assert np.allclose(m.y, y)

def test_clear_resets():
    m = Swish()
    m.train()
    x = np.array([-1.0, 0.0, 1.0])
    m.forward(x)
    dy = np.array([1.0, 2.0, 3.0])
    m.backward(dy)
    m.clear()
    assert m.x is None
    assert m.y is None
    assert m.dy is None