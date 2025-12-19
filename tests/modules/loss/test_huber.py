import numpy as np
import pytest
from deepi.modules.loss import Huber

@pytest.mark.parametrize("delta", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("reduction", [None, "sum", "mean"])
def test_forward_exact(delta, reduction):
    loss_fn = Huber(delta=delta, reduction=reduction)
    loss_fn.train()

    y_pred = np.array([[2.0, 0.0, -1.0],
                       [0.5, 1.5, -0.5]])
    y_true = np.array([[1.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0]])

    diff = y_pred - y_true
    abs_diff = np.abs(diff)
    mask = abs_diff <= delta
    loss_elem = mask * 0.5 * diff**2 + (~mask) * (delta * (abs_diff - 0.5 * delta))
    per_sample = loss_elem.reshape(loss_elem.shape[0], -1).sum(axis=1)

    loss = loss_fn(y_pred, y_true)

    if reduction is None:
        expected = per_sample
    elif reduction == "sum":
        expected = per_sample.sum()
    else:
        expected = per_sample.mean()

    assert np.allclose(loss, expected), f"Forward output mismatch with delta={delta}, reduction={reduction}"

@pytest.mark.parametrize("delta", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("reduction", [None, "sum", "mean"])
def test_backward_exact(delta, reduction):
    loss_fn = Huber(delta=delta, reduction=reduction)
    loss_fn.train()

    y_pred = np.array([[2.0, 0.0, -1.0],
                       [0.5, 1.5, -0.5]])
    y_true = np.array([[1.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0]])

    loss_fn(y_pred, y_true)
    grad = loss_fn.backward()

    diff = y_pred - y_true
    abs_diff = np.abs(diff)
    mask = abs_diff <= delta
    expected = mask * diff + (~mask) * (delta * np.sign(diff))

    if reduction == "mean":
        expected /= y_pred.shape[0]  # divide by batch size

    assert np.allclose(grad, expected), f"Backward gradients mismatch with delta={delta}, reduction={reduction}"

def test_forward_eval_mode_no_cache():
    loss_fn = Huber()
    loss_fn.eval()

    y_pred = np.array([[1.0, -0.5]])
    y_true = np.array([[0.0, 0.0]])

    loss = loss_fn(y_pred, y_true)
    assert loss_fn.x is None
    assert loss_fn.y is None

def test_forward_train_mode_caches():
    loss_fn = Huber()
    loss_fn.train()

    y_pred = np.array([[1.0, -0.5]])
    y_true = np.array([[0.0, 0.0]])

    loss = loss_fn(y_pred, y_true)
    assert np.allclose(loss_fn.x[0], y_pred)
    assert np.allclose(loss_fn.x[1], y_true)
    assert np.allclose(loss_fn.y, loss)

def test_clear_resets():
    loss_fn = Huber()
    loss_fn.train()

    y_pred = np.array([[1.0, -0.5]])
    y_true = np.array([[0.0, 0.0]])

    loss_fn(y_pred, y_true)
    loss_fn.backward()
    loss_fn.clear()

    assert loss_fn.x is None
    assert loss_fn.y is None
    assert loss_fn.dy is None