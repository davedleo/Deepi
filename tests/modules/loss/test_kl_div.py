import numpy as np
import pytest
from deepi.modules.loss import KLDiv

# --------------------------------------------------------------------------
# Tests for KLDiv loss
# --------------------------------------------------------------------------

@pytest.mark.parametrize("reduction", [None, "sum", "mean"])
def test_forward_exact(reduction):
    loss_fn = KLDiv(reduction=reduction)
    loss_fn.train()

    y_hat = np.array([[0.6, 0.3, 0.1],
                      [0.3, 0.4, 0.3]])
    y = np.array([[0.5, 0.4, 0.1],
                  [0.2, 0.5, 0.3]])

    eps = loss_fn.eps

    loss = loss_fn(y_hat, y)

    # KL: sum_i y_i (log(y_i) - log(y_hat_i))
    elementwise = y * (np.log(y + eps) - np.log(y_hat + eps))
    per_sample_sum = elementwise.sum(axis=1)  # reduction before top-level reduction

    if reduction is None:
        expected = per_sample_sum
    elif reduction == "sum":
        expected = per_sample_sum.sum()
    else:  # mean
        expected = per_sample_sum.mean()

    assert np.allclose(loss, expected), f"Forward output mismatch with reduction={reduction}"


@pytest.mark.parametrize("reduction", [None, "sum", "mean"])
def test_backward_exact(reduction):
    loss_fn = KLDiv(reduction=reduction)
    loss_fn.train()

    y_hat = np.array([[0.6, 0.3, 0.1],
                      [0.3, 0.4, 0.3]])
    y     = np.array([[0.5, 0.4, 0.1],
                      [0.2, 0.5, 0.3]])

    eps = loss_fn.eps

    loss_fn(y_hat, y)
    grad = loss_fn.backward()

    # d/d(y_hat): - y / y_hat
    expected = - y / (y_hat + eps)

    if reduction == "mean":
        # The base Loss norm divides by number of samples
        expected /= y.shape[0]

    assert np.allclose(grad, expected), f"Backward gradients mismatch with reduction={reduction}"


def test_forward_eval_mode_no_cache():
    loss_fn = KLDiv()
    loss_fn.eval()

    y_hat = np.array([[0.6, 0.3, 0.1]])
    y     = np.array([[0.5, 0.4, 0.1]])

    loss = loss_fn(y_hat, y)
    assert loss_fn.x is None
    assert loss_fn.y is None


def test_forward_train_mode_caches():
    loss_fn = KLDiv()
    loss_fn.train()

    y_hat = np.array([[0.6, 0.3, 0.1]])
    y     = np.array([[0.5, 0.4, 0.1]])

    loss = loss_fn(y_hat, y)
    assert np.allclose(loss_fn.x[0], y_hat)
    assert np.allclose(loss_fn.x[1], y)
    assert np.allclose(loss_fn.y, loss)


def test_clear_resets():
    loss_fn = KLDiv()
    loss_fn.train()

    y_hat = np.array([[0.6, 0.3, 0.1]])
    y     = np.array([[0.5, 0.4, 0.1]])

    loss_fn(y_hat, y)
    loss_fn.backward()

    loss_fn.clear()
    assert loss_fn.x is None
    assert loss_fn.y is None
    assert loss_fn.dy is None