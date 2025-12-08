import numpy as np
import pytest
from deepi.modules.flow.flatten import Flatten

# --------------------------------------------------------------------------
# Tests for Flatten flow layer
# --------------------------------------------------------------------------

def test_forward_flatten_exact():
    m = Flatten()
    x = np.array([[[1., 2.], [3., 4.]]])  # (1,2,2)
    y = m.forward(x)
    expected = np.array([[1., 2., 3., 4.]])
    assert np.allclose(y, expected), "Forward flatten result mismatch"


def test_backward_exact():
    m = Flatten()
    m.train()

    x = np.random.randn(2, 3, 4)  # (batch=2)
    y = m.forward(x)

    dy = np.random.randn(2, 12)
    dx = m.gradients(dy)

    # expected shape follows updated logic
    expected_shape = (2, 3, 4)
    assert dx.shape == expected_shape, "Reshaping mismatch after backward"
    assert np.allclose(dx.reshape(2, -1), dy), \
        "Gradient values should be identical after flatten + reshape"


def test_backward_accumulation():
    m = Flatten()
    m.train()

    x = np.zeros((4, 3, 2))  # (batch=4)
    m.forward(x)

    dy1 = np.ones((4, 6))
    dy2 = np.full((4, 6), 2.0)

    m.backward(dy1)
    m.backward(dy2)

    total_dy = dy1 + dy2
    dx = m.gradients(m.dy)

    expected_shape = (4, 3, 2)
    assert dx.shape == expected_shape, "Accumulated dx reshape incorrect"

    # verify flatten-reshape identity logic
    assert np.allclose(dx.reshape(4, -1), total_dy), \
        "Flatten backward accumulation incorrect"


def test_eval_mode_no_cache():
    m = Flatten()
    m.eval()
    x = np.random.randn(2, 5, 6)
    y = m.forward(x)

    assert m.x is None and m.y is None, "Cache should not store values in eval mode"


def test_train_mode_cache():
    m = Flatten()
    m.train()
    x = np.random.randn(3, 4, 2)
    y = m.forward(x)

    assert np.allclose(m.x, x)
    assert np.allclose(m.y, y), "Training cache values incorrect"


def test_clear_resets():
    m = Flatten()
    m.train()

    x = np.random.randn(2, 2, 3)
    m.forward(x)

    dy = np.random.randn(2, 6)
    m.backward(dy)

    m.clear()

    assert m.x is None
    assert m.y is None
    assert m.dy is None, "clear() must reset all buffers"