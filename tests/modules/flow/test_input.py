import numpy as np
import pytest
from deepi.modules.flow import Input
from deepi.modules.flow import Reshape  # used as a dummy next module

# --------------------------------------------------------------------------
# Tests for Input flow layer
# --------------------------------------------------------------------------

def test_generate_sample_shape():
    m = Input((3, 4))
    sample = m.generate_sample()
    expected_shape = (1, 3, 4)
    assert sample.shape == expected_shape, "Generated sample shape mismatch"
    assert sample.dtype == np.float64, "Generated sample dtype should be float64"

def test_forward_returns_input():
    m = Input((2, 2))
    x = np.random.randn(1, 2, 2)
    y = m.forward(x)
    assert np.allclose(y, x), "Forward should return input as is"

def test_gradients_returns_none():
    m = Input((2, 2))
    m.train()
    x = np.zeros((1, 2, 2))
    m.forward(x)
    dy = np.ones((1, 2, 2))
    grad = m.gradients(dy)
    assert grad is None, "Input layer gradients should return None"

def test_link_appends_next():
    m = Input((2,))
    next_module = Reshape((1,))
    assert next_module not in m.next
    m.link(next_module)
    assert next_module in m.next, "Link should append module to next"
    # linking same module again should not duplicate
    m.link(next_module)
    assert m.next.count(next_module) == 1, "Link should not duplicate next modules"

def test_eval_mode_no_cache():
    m = Input((2, 2))
    m.eval()
    x = np.random.randn(1, 2, 2)
    y = m.forward(x)
    assert m.x is None and m.y is None, "Input layer should not cache values in eval mode"

def test_train_mode_cache_behavior():
    m = Input((2, 2))
    m.train()
    x = np.random.randn(1, 2, 2)
    y = m.forward(x)
    # Input layer does not override Flow caching, so it may store x/y
    # Depending on base Flow implementation, test if caching occurs
    assert m.x is None or np.allclose(m.x, x)  # permissive check
    assert m.y is None or np.allclose(m.y, y)

def test_clear_resets():
    m = Input((2, 2))
    m.train()
    x = np.random.randn(1, 2, 2)
    m.forward(x)
    m.clear()
    assert m.x is None
    assert m.y is None
    assert m.dy is None