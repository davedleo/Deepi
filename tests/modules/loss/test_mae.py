from typing import Optional
import numpy as np
import pytest
from deepi.modules.loss import MAE

@pytest.mark.parametrize("reduction", [None, "sum", "mean"])
def test_forward_exact(reduction):
    loss_fn = MAE(reduction=reduction)
    loss_fn.train()

    y_hat = np.array([[1.5, 1.5, 3.0],
                      [2.0, 0.5, 1.0]])
    y = np.array([[1.0, 2.0, 3.0],
                  [2.0, 1.0, 0.0]])

    loss = loss_fn.forward(y_hat, y)

    elementwise = np.abs(y_hat - y)
    per_sample_mean = elementwise.mean(axis=1)

    if reduction is None:
        expected = per_sample_mean
    elif reduction == "sum":
        expected = per_sample_mean.sum()
    else:  # mean
        expected = per_sample_mean.mean()

    assert np.allclose(loss, expected), f"Forward output mismatch with reduction={reduction}"

@pytest.mark.parametrize("reduction", [None, "sum", "mean"])
def test_backward_exact(reduction):
    loss_fn = MAE(reduction=reduction)
    loss_fn.train()

    y_hat = np.array([[1.5, 1.5, 3.0],
                      [2.0, 0.5, 1.0]])
    y = np.array([[1.0, 2.0, 3.0],
                  [2.0, 1.0, 0.0]])

    loss_fn.forward(y_hat, y)
    grad = loss_fn.backward()

    expected = np.sign(y_hat - y) / y_hat.shape[1]
    if reduction == "mean":
        expected /= y.shape[0]

    assert np.allclose(grad, expected), f"Backward gradients mismatch with reduction={reduction}"

def test_forward_eval_mode_no_cache():
    loss_fn = MAE()
    loss_fn.eval()

    y_hat = np.array([[1.5, 1.5, 3.0]])
    y = np.array([[1.0, 2.0, 3.0]])

    loss = loss_fn.forward(y_hat, y)
    assert loss_fn.x is None
    assert loss_fn.y is None

def test_forward_train_mode_caches():
    loss_fn = MAE()
    loss_fn.train()

    y_hat = np.array([[1.5, 1.5, 3.0]])
    y = np.array([[1.0, 2.0, 3.0]])

    loss = loss_fn.forward(y_hat, y)
    assert np.allclose(loss_fn.x[0], y_hat)
    assert np.allclose(loss_fn.x[1], y)
    assert np.allclose(loss_fn.y, loss)

def test_clear_resets():
    loss_fn = MAE()
    loss_fn.train()

    y_hat = np.array([[1.5, 1.5, 3.0]])
    y = np.array([[1.0, 2.0, 3.0]])

    loss_fn.forward(y_hat, y)
    loss_fn.backward()

    loss_fn.clear()
    assert loss_fn.x is None
    assert loss_fn.y is None
    assert loss_fn.dy is None
