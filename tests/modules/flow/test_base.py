import numpy as np
import pytest
from deepi.modules.flow import Flow

# --------------------------------------------------------------------------
# Dummy flow for testing
# --------------------------------------------------------------------------

class DummyFlow(Flow):
    def __init__(self):
        super().__init__("dummy")

    def forward(self, x: np.ndarray) -> np.ndarray:
        # flatten only, no scaling
        return x.reshape(x.shape[0], -1)

    def gradients(self, dy: np.ndarray) -> np.ndarray:
        # reshape back to stored input shape
        return dy.reshape(self.x.shape)

# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

def test_forward_without_training():
    """Forward should NOT cache x,y if not in training mode"""
    m = DummyFlow()
    x = np.array([[1., 2., 3.]])
    y = m(x)

    # flatten should not change shape because already 2-D
    assert np.allclose(y, x.reshape(1, 3))
    assert m.x is None
    assert m.y is None


def test_forward_with_training_caching():
    """Forward must cache x,y when training flag is set"""
    m = DummyFlow()
    m.train()
    x = np.array([[1., 3.]])
    y = m(x)
    assert np.allclose(y, x.reshape(1, 2))

    # now caching should be active
    assert np.allclose(m.x, x)
    assert np.allclose(m.y, y)


def test_backward_accumulation():
    """Backward should accumulate gradients in m.dy"""
    m = DummyFlow()
    m.train()

    # set input shape so backward can reshape
    m(np.zeros((1, 3)))

    dy1 = np.array([[1., 2., 3.]])
    dy2 = np.array([[2., 3., 4.]])

    m.backward(dy1)
    m.backward(dy2)

    # original arrays untouched
    assert dy1.shape == (1, 3)
    assert dy2.shape == (1, 3)

    # accumulation
    assert np.allclose(m.dy, dy1 + dy2)

    # gradients call must reshape back to original x shape
    # set input shape (as if forward was called)
    m.x = np.zeros((1, 3))
    dx = m.gradients(m.dy)
    assert np.allclose(dx, (dy1 + dy2).reshape(1, 3))


def test_clear_resets():
    """Clear should remove all cached state"""
    m = DummyFlow()
    m.train()
    x = np.array([[1., 2.]])
    m(x)
    m.backward(np.array([[3., 4.]]))

    assert m.x is not None
    assert m.y is not None
    assert m.dy is not None

    m.clear()

    assert m.x is None
    assert m.y is None
    assert m.dy is None