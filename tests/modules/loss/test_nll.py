import numpy as np
import pytest
from deepi.modules.loss import NLL


# --------------------------------------------------------------------------
# Tests for NLL Loss
# --------------------------------------------------------------------------


@pytest.mark.parametrize("reduction", [None, "sum", "mean"])
def test_forward_exact(reduction):
    loss_fn = NLL(reduction=reduction)
    loss_fn.train()

    # Values are ALREADY log-probabilities
    y_hat = np.log(np.array([
        [0.6, 0.3, 0.1],
        [0.3, 0.4, 0.3]
    ]))

    y = np.array([0, 2])  # correct classes

    loss = loss_fn.forward(y_hat, y)

    # NLL = -log p(correct_class)
    expected_losses = -y_hat[np.arange(2), y]

    if reduction is None:
        expected = expected_losses
    elif reduction == "sum":
        expected = expected_losses.sum()
    else:  # mean
        expected = expected_losses.mean()

    assert np.allclose(loss, expected), f"Forward mismatch with reduction={reduction}"


@pytest.mark.parametrize("reduction", [None, "sum", "mean"])
def test_backward_exact(reduction):
    loss_fn = NLL(reduction=reduction)
    loss_fn.train()

    y_hat = np.log(np.array([
        [0.6, 0.3, 0.1],
        [0.3, 0.4, 0.3]
    ]))

    y = np.array([0, 2])

    loss_fn.forward(y_hat, y)  # must call forward first
    grad = loss_fn.backward()

    # gradient is -1 at the correct class index, zero elsewhere
    expected = np.zeros_like(y_hat)
    expected[np.arange(2), y] = -1.0

    # reduction mean divides gradient by batch size
    if reduction == "mean":
        expected /= y_hat.shape[0]

    assert np.allclose(grad, expected), f"Backward mismatch with reduction={reduction}"


def test_forward_eval_no_cache():
    loss_fn = NLL()
    loss_fn.eval()

    y_hat = np.log(np.array([[0.6, 0.4]]))
    y = np.array([1])

    loss = loss_fn.forward(y_hat, y)

    assert loss_fn.x is None
    assert loss_fn.y is None


def test_forward_train_cached():
    loss_fn = NLL()
    loss_fn.train()

    y_hat = np.log(np.array([[0.6, 0.4]]))
    y = np.array([1])

    loss = loss_fn.forward(y_hat, y)

    assert np.allclose(loss_fn.x[0], y_hat)
    assert np.allclose(loss_fn.x[1], y)
    assert np.allclose(loss_fn.y, loss)


def test_clear_resets_state():
    loss_fn = NLL()
    loss_fn.train()

    y_hat = np.log(np.array([[0.6, 0.4]]))
    y = np.array([1])

    loss_fn.forward(y_hat, y)
    loss_fn.backward()

    loss_fn.clear()

    assert loss_fn.x is None
    assert loss_fn.y is None
    assert loss_fn.dy is None