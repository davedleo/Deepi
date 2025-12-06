import numpy as np
import pytest
from deepi.modules.loss.base import Loss

# --------------------------------------------------------------------------
# Dummy loss for testing
# --------------------------------------------------------------------------
class DummyLoss(Loss):
    def __init__(self, reduction="mean"):
        super().__init__("dummy", reduction=reduction)

    def loss_transform(self, y, y_hat):
        # cache inputs in self.x
        self.x = (y, y_hat)
        # compute elementwise squared difference
        loss = (y_hat - y) ** 2
        self.y = self.apply_reduction(loss)
        return self.y

    def loss_gradient(self):
        y, y_hat = self.x
        grad = 2 * (y_hat - y)
        return grad


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------
@pytest.mark.parametrize("reduction", [None, "sum", "mean"])
def test_forward_and_reduction(reduction):
    m = DummyLoss(reduction=reduction)
    y = np.array([1.0, 2.0, 3.0])
    y_hat = np.array([1.5, 1.5, 3.0])

    loss = m.transform(y, y_hat)

    elementwise = (y_hat - y) ** 2
    if reduction is None:
        expected = elementwise
    elif reduction == "sum":
        expected = elementwise.sum(keepdims=True)
    else:  # mean
        expected = elementwise.mean(keepdims=True)

    assert np.allclose(loss, expected)
    assert m.x == (y, y_hat)
    assert np.allclose(m.y, expected)


@pytest.mark.parametrize("reduction", [None, "sum", "mean"])
def test_gradients_with_dy(reduction):
    m = DummyLoss(reduction=reduction)
    y = np.array([1.0, 2.0, 3.0])
    y_hat = np.array([1.5, 1.5, 3.0])
    m.transform(y, y_hat)

    dy = np.ones_like(y_hat)
    grad = m.gradients(dy)

    base_grad = 2 * (y_hat - y) * dy
    if reduction == "mean":
        base_grad /= y.shape[0]

    assert np.allclose(grad, base_grad)


def test_clear_cache_resets():
    m = DummyLoss(reduction="mean")
    y = np.array([1.0, 2.0])
    y_hat = np.array([2.0, 3.0])
    m.transform(y, y_hat)
    m.clear()
    assert m.x is None
    assert m.y is None