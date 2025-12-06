import numpy as np
import pytest
from deepi.modules.loss.base import Loss

# --------------------------------------------------------------------------
# Dummy loss for testing
# --------------------------------------------------------------------------

class DummyLoss(Loss):
    def __init__(self, reduction=None):
        super().__init__("dummy", reduction=reduction)

    def transform(self, y, y_hat):
        # element-wise squared error
        return (y_hat - y) ** 2

    def gradients(self):
        y, y_hat = self.x
        return 2 * (y_hat - y)

# --------------------------------------------------------------------------
# Forward tests
# --------------------------------------------------------------------------

@pytest.mark.parametrize("reduction", [None, "sum", "mean"])
def test_forward_exact(reduction):
    m = DummyLoss(reduction=reduction)
    m.train()
    y = np.array([[1.0, 2.0, 3.0]])
    y_hat = np.array([[1.5, 1.5, 3.0]])

    loss = m.forward(y, y_hat)

    elementwise = (y_hat - y) ** 2
    if reduction is None:
        expected = elementwise
    elif reduction == "sum":
        expected = np.array([[elementwise.sum()]])
    else:  # mean
        expected = np.array([[elementwise.mean()]])

    assert np.allclose(loss, expected)
    assert np.allclose(m.x[0], y)
    assert np.allclose(m.x[1], y_hat)
    assert np.allclose(m.y, expected)

# --------------------------------------------------------------------------
# Backward tests
# --------------------------------------------------------------------------

@pytest.mark.parametrize("reduction", [None, "sum", "mean"])
def test_backward_exact(reduction):
    m = DummyLoss(reduction=reduction)
    m.train()
    y = np.array([[1.0, 2.0, 3.0]])
    y_hat = np.array([[1.5, 1.5, 3.0]])
    m.forward(y, y_hat)

    grad = m.backward()

    expected = 2 * (y_hat - y)
    if reduction == "mean":
        expected /= y.shape[0]

    assert np.allclose(grad, expected)
    assert np.allclose(m.dy, expected)

# --------------------------------------------------------------------------
# Backward accumulation
# --------------------------------------------------------------------------

def test_backward_accumulation():
    m = DummyLoss(reduction="mean")
    m.train()
    y = np.array([[1.0, 2.0]])
    y_hat = np.array([[2.0, 1.0]])
    m.forward(y, y_hat)

    grad1 = m.backward()
    grad2 = m.backward()

    expected_single = 2 * (y_hat - y) / y.shape[0]

    # dy should be overwritten on each backward call, not accumulated
    assert np.allclose(m.dy, expected_single)
    assert np.allclose(grad2, expected_single)

# --------------------------------------------------------------------------
# Train/eval mode caching
# --------------------------------------------------------------------------

def test_train_eval_mode_caching():
    y = np.array([[1.0, 2.0]])
    y_hat = np.array([[2.0, 3.0]])

    m = DummyLoss(reduction="mean")
    m.train()
    m.forward(y, y_hat)
    assert m.x is not None
    assert m.y is not None

    m.clear()  # clear cache before switching to eval
    m.eval()
    m.forward(y, y_hat)
    assert m.x is None
    assert m.y is None

# --------------------------------------------------------------------------
# Clear/reset
# --------------------------------------------------------------------------

def test_clear_resets():
    y = np.array([[1.0, 2.0]])
    y_hat = np.array([[2.0, 3.0]])

    m = DummyLoss(reduction="mean")
    m.train()
    m.forward(y, y_hat)
    m.backward()

    m.clear()
    assert m.x is None
    assert m.y is None
    assert m.dy is None