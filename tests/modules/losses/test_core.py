import pytest
import numpy as np
from deepi.modules.losses import (
    Loss,
    CrossEntropy,
    ElasticNet,
    GaussianNLL,
    KLDiv,
    MAE,
    ModifiedUber,
    MSE,
    NLL,
    PoissonNLL,
)

# Fixtures
@pytest.fixture
def y():
    return np.array([0, 1, 2, 1, 0])

@pytest.fixture
def y_hat_classification():
    return np.array([[0.2, 0.3, 0.5],
                     [0.1, 0.8, 0.1],
                     [0.7, 0.2, 0.1],
                     [0.05, 0.9, 0.05],
                     [0.6, 0.2, 0.2]])

@pytest.fixture
def y_hat_regression():
    return np.array([0.1, 1.2, 1.9, 0.8, -0.1])

@pytest.fixture
def dy():
    return np.ones(5) * 0.1


# --- Dummy subclass to test base Loss ---
class DummyLoss(Loss):
    def forward(self, y, y_hat):
        if self._is_training:
            self.dx = np.ones_like(y_hat) * 0.5
        return np.sum(y_hat - y)


def test_dummy_loss(y, y_hat_regression, dy):
    loss = DummyLoss("dummy")
    
    # Initialization
    assert loss.type == "module.loss.dummy"
    assert not loss._is_training
    assert loss.dx == 0.0

    # Forward no training
    out = loss.forward(y, y_hat_regression)
    assert isinstance(out, np.ndarray) or np.isscalar(out)
    assert not loss._is_training
    assert loss.dx == 0.0

    # Forward training
    loss.train()
    out_train = loss.forward(y, y_hat_regression)
    assert loss._is_training
    assert np.all(loss.dx == 0.5)

    # Backward
    dx = loss.backward()
    assert isinstance(dx, np.ndarray)
    assert np.all(dx == 0.5)


# --- CrossEntropy ---
def test_cross_entropy(y, y_hat_classification):
    # Without weights
    ce = CrossEntropy()
    assert ce.type == "module.loss.cross_entropy"
    assert not ce._is_training

    # Forward no training
    loss_val = ce.forward(y, y_hat_classification)
    # Compute expected cross-entropy loss manually
    eps = 1e-12
    # Softmax computation as in original code
    exp_logits = np.exp(y_hat_classification - np.max(y_hat_classification, axis=1, keepdims=True))
    softmax = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    y_one_hot = np.zeros_like(y_hat_classification)
    y_one_hot[np.arange(len(y)), y] = 1
    clipped_preds = np.clip(softmax, eps, 1 - eps)
    expected_loss = -np.sum(y_one_hot * np.log(clipped_preds)) / len(y)
    assert np.allclose(loss_val, expected_loss, rtol=1e-5, atol=1e-8)
    assert ce.dx == 0.0

    # Forward training
    ce.train()
    loss_train = ce.forward(y, y_hat_classification)
    assert ce._is_training
    assert isinstance(ce.dx, np.ndarray)
    assert ce.dx.shape == y_hat_classification.shape
    # Expected gradient: (softmax - one_hot) / N
    expected_dx = (clipped_preds - y_one_hot) / len(y)
    assert np.allclose(ce.dx, expected_dx, rtol=1e-5, atol=1e-8)

    # Backward
    dx = ce.backward()
    assert dx.shape == y_hat_classification.shape


# --- ElasticNet ---
def test_elasticnet(y, y_hat_regression):
    en = ElasticNet(0.3, 0.7)
    assert en.type == "module.loss.elasticnet"

    # Forward no training
    out = en.forward(y, y_hat_regression)
    # Expected loss = 0.3 * L1 + 0.7 * L2 (without dividing L2 by 2)
    l1 = np.sum(np.abs(y_hat_regression - y)) / len(y)
    l2 = np.sum((y_hat_regression - y)**2) / len(y)
    expected_out = 0.3 * l1 + 0.7 * l2
    assert np.allclose(out, expected_out, rtol=1e-5, atol=1e-8)
    assert en.dx == 0.0

    # Forward training
    en.train()
    out_train = en.forward(y, y_hat_regression)
    assert np.all(en.dx.shape == y_hat_regression.shape)
    # Expected gradient: 0.3 * sign + 0.7 * 2 * diff / N
    diff = y_hat_regression - y
    expected_dx = 0.3 * np.sign(diff) / len(y) + 0.7 * (2 * diff / len(y))
    assert np.allclose(en.dx, expected_dx, rtol=1e-5, atol=1e-8)

    # Backward
    dx = en.backward()
    assert dx.shape == y_hat_regression.shape


# --- GaussianNLL ---
def test_gaussian_nll(y, y_hat_regression):
    gn = GaussianNLL()
    assert gn.type == "module.loss.gaussian_nll"

    eps = 1e-12
    diff = y - y_hat_regression
    var = np.var(y_hat_regression) + eps
    expected_out = 0.5 * ((diff ** 2) / var + np.log(2 * np.pi * var)).mean()

    out = gn.forward(y, y_hat_regression)
    assert np.allclose(out, expected_out, rtol=1e-5, atol=1e-8)

    gn.train()
    out_train = gn.forward(y, y_hat_regression)
    assert gn.dx.shape == y_hat_regression.shape

    N = len(y)
    expected_dx = diff / var / N
    assert np.allclose(gn.dx, expected_dx, rtol=1e-5, atol=1e-8)

    dx = gn.backward()
    assert dx.shape == y_hat_regression.shape


# --- KLDiv ---
def test_kldiv(y_hat_regression):
    # y and y_hat should be probability distributions
    y_dist = np.clip(y_hat_regression, 1e-3, 1.0)
    y_dist /= y_dist.sum()
    kld = KLDiv()
    # Per-sample normalization: divide by number of elements (len(y_dist))
    out = kld.forward(y_dist, y_dist)
    # KL divergence of distribution with itself is zero (per-sample normalization)
    expected_out = 0.0
    assert np.allclose(out, expected_out, rtol=1e-5, atol=1e-8)

    kld.train()
    kld.forward(y_dist, y_dist)
    assert kld.dx.shape == y_dist.shape
    # Gradient = -p / q / N, here p=q, so gradient = -1 / N for each element
    expected_dx = -np.ones_like(y_dist) / len(y_dist)
    assert np.allclose(kld.dx, expected_dx, rtol=1e-5, atol=1e-8)

    dx = kld.backward()
    assert dx.shape == y_dist.shape


# --- MAE ---
def test_mae(y, y_hat_regression):
    mae = MAE()
    out = mae.forward(y, y_hat_regression)
    # Mean absolute error
    expected_out = np.sum(np.abs(y_hat_regression - y)) / len(y)
    assert np.allclose(out, expected_out, rtol=1e-5, atol=1e-8)

    mae.train()
    mae.forward(y, y_hat_regression)
    assert mae.dx.shape == y_hat_regression.shape
    # Gradient = sign(y_hat - y) / N
    expected_dx = np.sign(y_hat_regression - y) / len(y)
    assert np.allclose(mae.dx, expected_dx, rtol=1e-5, atol=1e-8)

    dx = mae.backward()
    assert dx.shape == y_hat_regression.shape


# --- ModifiedUber ---
def test_modified_uber(y, y_hat_regression):
    mu = ModifiedUber()
    out = mu.forward(y, y_hat_regression)
    # ModifiedUber loss is assumed similar to MSE scaled by 0.5 (assuming)
    expected_out = np.sum((y_hat_regression - y) ** 2) / (2 * len(y))
    assert np.allclose(out, expected_out, rtol=1e-5, atol=1e-8)

    mu.train()
    mu.forward(y, y_hat_regression)
    assert mu.dx.shape == y_hat_regression.shape
    # Gradient as per original code's piecewise formula
    diff = y_hat_regression - y
    dx_expected = np.zeros_like(diff)
    N = len(y)
    for i in range(N):
        d = diff[i]
        if d < -1:
            dx_expected[i] = -1 / N
        elif -1 <= d <= 1:
            dx_expected[i] = d / N
        else:  # d > 1
            dx_expected[i] = 1 / N
    assert np.allclose(mu.dx, dx_expected, rtol=1e-5, atol=1e-8)

    dx = mu.backward()
    assert dx.shape == y_hat_regression.shape


# --- MSE ---
def test_mse(y, y_hat_regression):
    mse = MSE()
    out = mse.forward(y, y_hat_regression)
    # Mean squared error (mean of squared differences)
    diff = y_hat_regression - y
    expected_out = np.mean(diff ** 2)
    assert np.allclose(out, expected_out, rtol=1e-5, atol=1e-8)

    mse.train()
    mse.forward(y, y_hat_regression)
    assert mse.dx.shape == y_hat_regression.shape
    # Gradient = 2 * diff / N
    expected_dx = 2 * diff / len(y)
    assert np.allclose(mse.dx, expected_dx, rtol=1e-5, atol=1e-8)

    dx = mse.backward()
    assert dx.shape == y_hat_regression.shape


# --- NLL ---
def test_nll(y_hat_regression):
    y_target = np.ones_like(y_hat_regression) * 0.5
    nll = NLL()
    out = nll.forward(y_target, y_hat_regression)
    # Negative log likelihood assuming Gaussian with mean y_hat_regression and target y_target
    # forward = -(y * log(clipped)).sum() / N
    eps = 1e-12
    clipped = np.clip(y_hat_regression, eps, 1.0)
    expected_out = -(y_target * np.log(clipped)).sum() / len(y_target)
    assert np.allclose(out, expected_out, rtol=1e-5, atol=1e-8)

    nll.train()
    nll.forward(y_target, y_hat_regression)
    assert nll.dx.shape == y_hat_regression.shape
    # Gradient = -y / clipped / N
    expected_dx = -y_target / clipped / len(y_target)
    assert np.allclose(nll.dx, expected_dx, rtol=1e-5, atol=1e-8)

    dx = nll.backward()
    assert dx.shape == y_hat_regression.shape


# --- PoissonNLL ---
def test_poisson_nll(y_hat_regression):
    y_target = np.ones_like(y_hat_regression)
    pn = PoissonNLL()
    out = pn.forward(y_target, y_hat_regression)
    # Poisson negative log likelihood = (clipped - y * log(clipped)).mean()
    eps = 1e-12
    clipped_y_hat = np.clip(y_hat_regression, eps, None)
    expected_out = np.mean(clipped_y_hat - y_target * np.log(clipped_y_hat))
    assert np.allclose(out, expected_out, rtol=1e-5, atol=1e-8)

    pn.train()
    pn.forward(y_target, y_hat_regression)
    assert pn.dx.shape == y_hat_regression.shape
    # Gradient = (1 - y_target / clipped) / N to match per-sample normalization
    expected_dx = (1 - y_target / clipped_y_hat) / len(y_target)
    assert np.allclose(pn.dx, expected_dx, rtol=1e-5, atol=1e-8)

    dx = pn.backward()
    assert dx.shape == y_hat_regression.shape
