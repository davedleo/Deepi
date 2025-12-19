import numpy as np
import pytest
from deepi.modules.loss import ElasticNet

@pytest.mark.parametrize("reduction", [None, "sum", "mean"])
@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
def test_forward_exact(reduction, alpha):
    loss_fn = ElasticNet(alpha=alpha, reduction=reduction)
    loss_fn.train()

    y_hat = np.array([[1.0, 2.0, 3.0],
                      [2.0, 0.5, 1.0]])
    y = np.array([[0.0, 1.0, 2.0],
                  [2.0, 1.0, 0.0]])

    loss = loss_fn.forward(y_hat, y)

    diff = y_hat - y
    l1 = np.abs(diff).sum(axis=1)
    l2 = (diff ** 2).sum(axis=1)
    per_sample = (1.0 - alpha) * l1 + alpha * l2

    if reduction is None:
        expected = per_sample
    elif reduction == "sum":
        expected = per_sample.sum()
    else:  # mean
        expected = per_sample.mean()

    assert np.allclose(loss, expected), f"Forward output mismatch for alpha={alpha}, reduction={reduction}"


@pytest.mark.parametrize("reduction", [None, "sum", "mean"])
@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
def test_backward_exact(reduction, alpha):
    loss_fn = ElasticNet(alpha=alpha, reduction=reduction)
    loss_fn.train()

    y_hat = np.array([[1.0, 2.0, 3.0],
                      [2.0, 0.5, 1.0]])
    y = np.array([[0.0, 1.0, 2.0],
                  [2.0, 1.0, 0.0]])

    loss_fn.forward(y_hat, y)
    grad = loss_fn.backward()

    diff = y_hat - y
    grad_l1 = np.sign(diff)
    grad_l2 = 2.0 * diff
    expected = (1.0 - alpha) * grad_l1 + alpha * grad_l2

    if reduction == "mean":
        expected /= y_hat.shape[0]

    assert np.allclose(grad, expected), f"Backward gradients mismatch for alpha={alpha}, reduction={reduction}"


def test_forward_eval_mode_no_cache():
    loss_fn = ElasticNet()
    loss_fn.eval()

    y_hat = np.array([[1.0, 2.0]])
    y = np.array([[2.0, 3.0]])

    loss = loss_fn.forward(y_hat, y)
    assert loss_fn.x is None
    assert loss_fn.y is None


def test_forward_train_mode_caches():
    loss_fn = ElasticNet(alpha=0.5)
    loss_fn.train()

    y_hat = np.array([[1.0, 2.0]])
    y = np.array([[2.0, 3.0]])

    loss = loss_fn.forward(y_hat, y)
    assert np.allclose(loss_fn.x[0], y_hat)
    assert np.allclose(loss_fn.x[1], y)
    assert np.allclose(loss_fn.y, loss)


def test_clear_resets():
    loss_fn = ElasticNet()
    loss_fn.train()

    y_hat = np.array([[1.0, 2.0]])
    y = np.array([[2.0, 3.0]])

    loss_fn.forward(y_hat, y)
    loss_fn.backward()

    loss_fn.clear()
    assert loss_fn.x is None
    assert loss_fn.y is None
    assert loss_fn.dy is None