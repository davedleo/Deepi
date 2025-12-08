import numpy as np
import pytest
from deepi.modules.flow.reshape import Reshape

# --------------------------------------------------------------------------
# Tests for Reshape flow layer
# --------------------------------------------------------------------------

def test_forward_exact():
    """Check forward reshape correctness."""
    m = Reshape((2, 2))  # target shape excluding batch
    x = np.arange(4).reshape(1, 4)  # starting shape (1,4)
    y = m.forward(x)
    
    expected = np.arange(4).reshape(1, 2, 2)
    assert np.allclose(y, expected), "Forward reshape result mismatch"


def test_backward_exact():
    """Check gradients reshaping back to original input shape."""
    m = Reshape((4, 1))
    m.train()

    x = np.arange(8).reshape(2, 4)         # original shape (2,4)
    y = m.forward(x)                       # becomes (2,4,1)
    dy = np.ones((2, 4, 1))                # incoming gradient
    dx = m.gradients(dy)

    expected_shape = (2, 4)
    assert dx.shape == expected_shape, "Backward reshape incorrect"

    # flatten check
    assert np.allclose(dx, np.ones((2, 4))), "Backward values incorrect"


def test_backward_accumulation():
    """Check accumulation of backprop gradients."""
    m = Reshape((3, 2))
    m.train()

    x = np.zeros((5, 6))  # batch = 5
    m.forward(x)

    dy1 = np.ones((5, 3, 2))
    dy2 = np.full((5, 3, 2), 2.0)

    m.backward(dy1)
    m.backward(dy2)

    total_dy = dy1 + dy2
    dx = m.gradients(m.dy)

    expected_shape = (5, 6)

    assert dx.shape == expected_shape, "Accumulated dx reshape mismatch"
    assert np.allclose(dx.reshape(5, 6), total_dy.reshape(5, 6)), \
        "Accumulated values incorrect"


def test_eval_mode_no_cache():
    """Ensure eval mode does not store state."""
    m = Reshape((2, 2))
    m.eval()

    x = np.ones((4, 4))
    y = m.forward(x)

    assert m.x is None and m.y is None, \
        "Eval mode should not cache values"


def test_train_mode_cache():
    """Ensure train mode stores state."""
    m = Reshape((2, 2))
    m.train()

    x = np.arange(4).reshape(1, 4)
    y = m.forward(x)

    assert np.allclose(m.x, x)
    assert np.allclose(m.y, y)


def test_clear_resets():
    """Verify that clear empties buffers."""
    m = Reshape((2, 2))
    m.train()

    x = np.arange(4).reshape(1, 4)
    m.forward(x)

    dy = np.ones((1, 2, 2))
    m.backward(dy)

    m.clear()

    assert m.x is None
    assert m.y is None
    assert m.dy is None