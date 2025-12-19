import numpy as np
import pytest
from deepi.modules.loss import MSE

# --------------------------------------------------------------------------
# Tests for MSE loss
# --------------------------------------------------------------------------

@pytest.mark.parametrize("reduction", [None, "sum", "mean"])
def test_forward_exact(reduction):
    loss_fn = MSE(reduction=reduction)
    loss_fn.train()

    y_hat = np.array([[1.5, 1.5, 3.0],
                      [2.0, 0.5, 1.0]])
    y = np.array([[1.0, 2.0, 3.0],
                  [2.0, 1.0, 0.0]])

    loss = loss_fn(y_hat, y)

    elementwise = (y_hat - y) ** 2
    per_sample_mean = elementwise.mean(axis=1)  # 1D array: mean over features

    if reduction is None:
        expected = per_sample_mean
    elif reduction == "sum":
        expected = per_sample_mean.sum()  # scalar
    else:  # mean
        expected = per_sample_mean.mean()  # scalar

    assert np.allclose(loss, expected), f"Forward output mismatch with reduction={reduction}"


@pytest.mark.parametrize("reduction", [None, "sum", "mean"])
def test_backward_exact(reduction):
    loss_fn = MSE(reduction=reduction)
    loss_fn.train()

    y_hat = np.array([[1.5, 1.5, 3.0],
                      [2.0, 0.5, 1.0]])
    y = np.array([[1.0, 2.0, 3.0],
                  [2.0, 1.0, 0.0]])

    loss_fn(y_hat, y)
    grad = loss_fn.backward()

    expected = 2.0 * (y_hat - y) / 3
    if reduction == "mean":
        expected /= y.shape[0]  # normalize across samples

    assert np.allclose(grad, expected), f"Backward gradients mismatch with reduction={reduction}"


def test_forward_eval_mode_no_cache():
    loss_fn = MSE()
    loss_fn.eval()

    y_hat = np.array([[1.5, 1.5, 3.0]])
    y = np.array([[1.0, 2.0, 3.0]])

    loss = loss_fn(y_hat, y)
    assert loss_fn.x is None
    assert loss_fn.y is None


def test_forward_train_mode_caches():
    loss_fn = MSE()
    loss_fn.train()

    y_hat = np.array([[1.5, 1.5, 3.0]])
    y = np.array([[1.0, 2.0, 3.0]])

    loss = loss_fn(y_hat, y)
    assert np.allclose(loss_fn.x[0], y_hat)
    assert np.allclose(loss_fn.x[1], y)
    assert np.allclose(loss_fn.y, loss)


def test_clear_resets():
    loss_fn = MSE()
    loss_fn.train()

    y_hat = np.array([[1.5, 1.5, 3.0]])
    y = np.array([[1.0, 2.0, 3.0]])

    loss_fn(y_hat, y)
    loss_fn.backward()

    loss_fn.clear()
    assert loss_fn.x is None
    assert loss_fn.y is None
    assert loss_fn.dy is None