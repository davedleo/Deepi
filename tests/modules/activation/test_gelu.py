import numpy as np
import pytest
from scipy.special import erf
from deepi.modules.activation.gelu import GELU

# --------------------------------------------------------------------------
# Tests for GELU activation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("approximate", [True, False])
def test_forward_exact(approximate):
    m = GELU(approximate=approximate)
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    y = m.forward(x)
    if approximate:
        x3 = x ** 3
        inner = np.sqrt(2 / np.pi) * (x + 0.044715 * x3)
        expected = 0.5 * x * (1.0 + np.tanh(inner))
    else:
        expected = 0.5 * x * (1.0 + erf(x / np.sqrt(2)))
    assert np.allclose(y, expected), "Forward output mismatch with exact formula"

@pytest.mark.parametrize("approximate", [True, False])
def test_backward_exact(approximate):
    m = GELU(approximate=approximate)
    m.train()
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    m.forward(x)
    dy = np.ones_like(x)
    dx = m.gradients(dy)
    if approximate:
        x2 = x ** 2
        x3 = x ** 3
        inner = np.sqrt(2 / np.pi) * (x + 0.044715 * x3)
        tanh_inner = np.tanh(inner)
        sech2 = 1 - tanh_inner ** 2
        d_inner_dx = np.sqrt(2 / np.pi) * (1.0 + 3.0 * 0.044715 * x2)
        expected_dx = 0.5 * (1.0 + tanh_inner + x * sech2 * d_inner_dx)
    else:
        erf_term = erf(x / np.sqrt(2))
        expected_dx = 0.5 * (1.0 + erf_term) + (x / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * x ** 2)
    assert np.allclose(dx, expected_dx), "Backward gradients mismatch with exact formula"

@pytest.mark.parametrize("approximate", [True, False])
def test_backward_accumulation(approximate):
    m = GELU(approximate=approximate)
    m.train()
    x = np.array([-1.0, 0.5, 2.0])
    m.forward(x)
    dy1 = np.array([1.0, 2.0, 3.0])
    dy2 = np.array([4.0, 5.0, 6.0])
    m.backward(dy1)
    m.backward(dy2)
    total_dy = dy1 + dy2
    dx = m.gradients(m.dy)
    if approximate:
        x2 = x ** 2
        x3 = x ** 3
        inner = np.sqrt(2 / np.pi) * (x + 0.044715 * x3)
        tanh_inner = np.tanh(inner)
        sech2 = 1 - tanh_inner ** 2
        d_inner_dx = np.sqrt(2 / np.pi) * (1.0 + 3.0 * 0.044715 * x2)
        expected_dx = total_dy * 0.5 * (1.0 + tanh_inner + x * sech2 * d_inner_dx)
    else:
        erf_term = erf(x / np.sqrt(2))
        expected_dx = total_dy * (0.5 * (1.0 + erf_term) + (x / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * x ** 2))
    assert np.allclose(dx, expected_dx), "Accumulated gradients incorrect"

@pytest.mark.parametrize("approximate", [True, False])
def test_eval_mode_no_cache(approximate):
    m = GELU(approximate=approximate)
    m.eval()
    x = np.array([-1.0, 0.0, 1.0])
    y = m.forward(x)
    assert m.x is None and m.y is None, "Cache should not store values in eval mode"

@pytest.mark.parametrize("approximate", [True, False])
def test_train_mode_cache(approximate):
    m = GELU(approximate=approximate)
    m.train()
    x = np.array([-1.0, 0.0, 1.0])
    y = m.forward(x)
    assert np.allclose(m.x, x)
    assert np.allclose(m.y, y)

@pytest.mark.parametrize("approximate", [True, False])
def test_clear_resets(approximate):
    m = GELU(approximate=approximate)
    m.train()
    x = np.array([-1.0, 0.0, 1.0])
    m.forward(x)
    dy = np.array([1.0, 2.0, 3.0])
    m.backward(dy)
    m.clear()
    assert m.x is None
    assert m.y is None
    assert m.dy is None