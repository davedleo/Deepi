import numpy as np
import pytest

from deepi import Model
from deepi.modules import Input, Dense, ReLU
from deepi.optimization.optimizers import SGD


# --------------------------------------------------------------------------
# Model builder
# --------------------------------------------------------------------------

def build_test_model():
    inp = Input((4,))
    dense1 = Dense(8)
    relu = ReLU()
    dense2 = Dense(3)

    inp.link(dense1)
    dense1.link(relu)
    relu.link(dense2)

    model = Model(inp, dense2)
    return model


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

def test_sgd_without_momentum():
    """SGD without momentum behaves like plain gradient descent"""
    model = build_test_model()
    model.train()

    opt = SGD(model, lr=0.1, momentum=0.0)

    for module in model.topology:
        if module.has_params:
            for k in module.params:
                module.grads[k] = np.ones_like(module.params[k])

    orig_params = model.get_params()
    opt.step()

    for name, module in model.modules.items():
        if module.has_params:
            for k, w in module.params.items():
                assert np.allclose(
                    w,
                    orig_params[name][k] - 0.1 * np.ones_like(w)
                )


def test_sgd_with_momentum_initializes_velocity():
    """Momentum SGD should initialize velocity buffers"""
    model = build_test_model()
    model.train()

    opt = SGD(model, lr=0.01, momentum=0.9)

    for module_buffer in opt.buffer["params"].values():
        for buf in module_buffer.values():
            assert "velocity" in buf
            assert isinstance(buf["velocity"], (float, np.ndarray))


def test_sgd_with_momentum_single_step():
    """First momentum step: v = mu * 0 + dw"""
    model = build_test_model()
    model.train()

    mu = 0.9
    opt = SGD(model, lr=0.1, momentum=mu)

    for module in model.topology:
        if module.has_params:
            for k in module.params:
                module.grads[k] = np.ones_like(module.params[k])

    orig_params = model.get_params()
    opt.step()

    # v = mu * 0 + dw = 1 → update = v
    for name, module in model.modules.items():
        if module.has_params:
            for k, w in module.params.items():
                expected = orig_params[name][k] - 0.1 * np.ones_like(w)
                assert np.allclose(w, expected)


def test_sgd_with_momentum_two_steps():
    """Momentum should accumulate across steps using PyTorch formula"""
    model = build_test_model()
    model.train()

    mu = 0.5
    opt = SGD(model, lr=0.1, momentum=mu)

    for module in model.topology:
        if module.has_params:
            for k in module.params:
                module.grads[k] = np.ones_like(module.params[k])

    orig_params = model.get_params()

    # step 1: v1 = mu * 0 + dw = 1, update1 = v1
    opt.step()

    # step 2: v2 = mu * v1 + dw = 0.5 * 1 + 1 = 1.5, update2 = v2
    opt.step()

    for name, module in model.modules.items():
        if module.has_params:
            for k, w in module.params.items():
                total_update = 0.1 * (1.0 + 1.5)
                expected = orig_params[name][k] - total_update * np.ones_like(w)
                assert np.allclose(w, expected)


def test_sgd_with_dampening():
    """Dampening scales gradient contribution (first step uses gradient directly)"""
    model = build_test_model()
    model.train()

    mu = 0.9
    tau = 0.5
    opt = SGD(model, lr=0.1, momentum=mu, dampening=tau)

    for module in model.topology:
        if module.has_params:
            for k in module.params:
                module.grads[k] = np.ones_like(module.params[k])

    orig_params = model.get_params()

    # Step 1: first step, velocity = gradient
    opt.step()
    first_update = 1.0  # v = 1

    # Step 2: velocity = mu * v_prev + (1 - tau) * dw = 0.9 * 1 + 0.5 * 1 = 1.4
    opt.step()
    second_update = 1.4  # v2
    total_update = 0.1 * first_update + 0.1 * second_update

    for name, module in model.modules.items():
        if module.has_params:
            for k, w in module.params.items():
                expected = orig_params[name][k] - total_update * np.ones_like(w)
                assert np.allclose(w, expected)


def test_sgd_with_nesterov():
    """Nesterov uses dw + mu * v (effect starts at second step)"""
    model = build_test_model()
    model.train()

    mu = 0.9
    opt = SGD(model, lr=0.1, momentum=mu, nesterov=True)

    for module in model.topology:
        if module.has_params:
            for k in module.params:
                module.grads[k] = np.ones_like(module.params[k])

    orig_params = model.get_params()

    # Step 1: velocity = gradient itself (no previous momentum)
    opt.step()
    # weights after first step: velocity = 1, update = v (no Nesterov effect yet)
    first_update = 1.0  # velocity = 1, update = v (no Nesterov on first step)

    # Step 2: velocity = mu * v_prev + dw = 0.9 * 1.0 + 1.0 = 1.9
    opt.step()
    # Nesterov update = dw + mu * v = 1.0 + 0.9 * 1.9 = 1.0 + 1.71 = 2.71
    # But the update applied is lr * (dw + mu * v)
    nesterov_update = 1.0 + mu * 1.9  # v in this step is 1.9
    total_update = 0.1 * first_update + 0.1 * nesterov_update

    for name, module in model.modules.items():
        if module.has_params:
            for k, w in module.params.items():
                expected = orig_params[name][k] - total_update * np.ones_like(w)
                assert np.allclose(w, expected)


def test_sgd_maximize_flag():
    """maximize=True inverts gradient direction"""
    model = build_test_model()
    model.train()

    opt = SGD(model, lr=0.1, momentum=0.0, maximize=True)

    for module in model.topology:
        if module.has_params:
            for k in module.params:
                module.grads[k] = np.ones_like(module.params[k])

    orig_params = model.get_params()
    opt.step()

    for name, module in model.modules.items():
        if module.has_params:
            for k, w in module.params.items():
                assert np.allclose(
                    w,
                    orig_params[name][k] + 0.1 * np.ones_like(w)
                )


def test_sgd_does_not_modify_gradients():
    """SGD must not modify gradients in-place"""
    model = build_test_model()
    model.train()

    opt = SGD(model, lr=0.1, momentum=0.9)

    grad_copies = {}
    for module in model.topology:
        if module.has_params:
            grad_copies[module] = {}
            for k in module.params:
                g = np.ones_like(module.params[k])
                module.grads[k] = g
                grad_copies[module][k] = g.copy()

    opt.step()

    for module in model.topology:
        if module.has_params:
            for k, g in module.grads.items():
                assert np.allclose(g, grad_copies[module][k])