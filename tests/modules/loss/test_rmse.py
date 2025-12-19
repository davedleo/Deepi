import numpy as np
import pytest
from deepi.modules.loss import RMSE

@pytest.mark.parametrize("reduction", [None, "sum", "mean"])
def test_forward_rmse(reduction):
    loss_fn = RMSE(reduction=reduction)
    loss_fn.train()
    y_hat = np.array([[1.5, 1.5, 3.0],
                      [2.0, 0.0, 1.0]])
    y = np.array([[1.0, 2.0, 3.0],
                  [2.0, 1.0, 1.0]])

    loss_fn.train()
    loss = loss_fn(y_hat, y)
    
    per_sample_mse = np.mean((y_hat - y) ** 2, axis=1)
    expected = np.sqrt(per_sample_mse + loss_fn.eps)
    if reduction == "sum":
        expected = expected.sum()
    elif reduction == "mean":
        expected = expected.mean()

    assert np.allclose(loss, expected), f"Forward output mismatch with reduction={reduction}"


@pytest.mark.parametrize("reduction", [None, "sum", "mean"])
def test_gradients_rmse(reduction):
    loss_fn = RMSE(reduction=reduction)
    loss_fn.train()
    y_hat = np.array([[1.5, 1.5, 3.0],
                      [2.0, 0.0, 1.0]])
    y = np.array([[1.0, 2.0, 3.0],
                  [2.0, 1.0, 1.0]])

    loss_fn.train()
    loss_fn(y_hat, y)
    grad = loss_fn.backward()

    diff = y_hat - y
    mse = np.mean(diff ** 2, axis=1, keepdims=True)
    expected_grad = diff / (np.sqrt(mse + loss_fn.eps) * diff.shape[1])
    if reduction == "sum":
        expected_grad = expected_grad
    elif reduction == "mean":
        expected_grad = expected_grad / y.shape[0]

    assert np.allclose(grad, expected_grad), f"Gradient mismatch with reduction={reduction}"


def test_backward_returns_array():
    loss_fn = RMSE(reduction="mean")
    loss_fn.train()
    y_hat = np.array([[1.0, 2.0]])
    y = np.array([[2.0, 1.0]])
    
    loss_fn(y_hat, y)  # forward must be called first
    grad = loss_fn.backward()  # backward should now work
    assert grad.shape == y.shape