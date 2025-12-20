import numpy as np
import pytest

from deepi import Model
from deepi.modules import Input, Dense, ReLU
from deepi.optimization.optimizers.sgd import SGD


# --------------------------------------------------------------------------
# Model builder for tests
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
    return model, inp, dense1, relu, dense2


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

def test_sgd_without_momentum():
    """SGD without momentum should behave like plain gradient descent"""
    model, inp, d1, r, d2 = build_test_model()
    model.train()

    opt = SGD(model, lr=0.1, momentum=0.0)

    for module in model.modules:
        if module.has_params:
            for k in module.params:
                module.grads[k] = np.ones_like(module.params[k])

    orig_params = model.get_params()
    opt.step()

    for name, module in model.modules_map.items():
        if module.has_params:
            for k, v in module.params.items():
                assert np.allclose(
                    v,
                    orig_params[name][k] - 0.1 * np.ones_like(v)
                )


def test_sgd_with_momentum_single_step():
    """Momentum buffer should accumulate gradient on first step"""
    model, inp, d1, r, d2 = build_test_model()
    model.train()

    mu = 0.9
    opt = SGD(model, lr=0.1, momentum=mu)

    for module in model.modules:
        if module.has_params:
            for k in module.params:
                module.grads[k] = np.ones_like(module.params[k])

    orig_params = model.get_params()
    opt.step()

    # v = mu * 0 + dw = 1
    # update = lr * v
    for name, module in model.modules_map.items():
        if module.has_params:
            for k, v in module.params.items():
                assert np.allclose(
                    v,
                    orig_params[name][k] - 0.1 * np.ones_like(v)
                )


def test_sgd_with_momentum_two_steps():
    """Momentum should accumulate across steps"""
    model, inp, d1, r, d2 = build_test_model()
    model.train()

    mu = 0.5
    opt = SGD(model, lr=0.1, momentum=mu)

    for module in model.modules:
        if module.has_params:
            for k in module.params:
                module.grads[k] = np.ones_like(module.params[k])

    orig_params = model.get_params()

    # step 1: v = 1
    opt.step()

    # step 2: v = mu * 1 + 1 = 1.5
    opt.step()

    for name, module in model.modules_map.items():
        if module.has_params:
            for k, v in module.params.items():
                expected = (
                    orig_params[name][k]
                    - 0.1 * 1.0
                    - 0.1 * 1.5
                )
                assert np.allclose(v, expected)


def test_sgd_with_dampening():
    """Dampening should scale the gradient contribution"""
    model, inp, d1, r, d2 = build_test_model()
    model.train()

    mu = 0.9
    tau = 0.5
    opt = SGD(model, lr=0.1, momentum=mu, dampening=tau)

    for module in model.modules:
        if module.has_params:
            for k in module.params:
                module.grads[k] = np.ones_like(module.params[k])

    orig_params = model.get_params()
    opt.step()

    # v = mu * 0 + (1 - tau) * dw = 0.5
    for name, module in model.modules_map.items():
        if module.has_params:
            for k, v in module.params.items():
                assert np.allclose(
                    v,
                    orig_params[name][k] - 0.1 * 0.5 * np.ones_like(v)
                )


def test_sgd_with_nesterov():
    """Nesterov should use dw + mu * v"""
    model, inp, d1, r, d2 = build_test_model()
    model.train()

    mu = 0.9
    opt = SGD(model, lr=0.1, momentum=mu, nesterov=True)

    for module in model.modules:
        if module.has_params:
            for k in module.params:
                module.grads[k] = np.ones_like(module.params[k])

    orig_params = model.get_params()
    opt.step()

    # v = 1
    # update = dw + mu * v = 1 + 0.9 = 1.9
    for name, module in model.modules_map.items():
        if module.has_params:
            for k, v in module.params.items():
                assert np.allclose(
                    v,
                    orig_params[name][k] - 0.1 * 1.9 * np.ones_like(v)
                )


def test_sgd_maximize_flag():
    """maximize=True should invert gradient sign"""
    model, inp, d1, r, d2 = build_test_model()
    model.train()

    opt = SGD(model, lr=0.1, momentum=0.0, maximize=True)

    for module in model.modules:
        if module.has_params:
            for k in module.params:
                module.grads[k] = np.ones_like(module.params[k])

    orig_params = model.get_params()
    opt.step()

    for name, module in model.modules_map.items():
        if module.has_params:
            for k, v in module.params.items():
                assert np.allclose(
                    v,
                    orig_params[name][k] + 0.1 * np.ones_like(v)
                )


def test_sgd_buffer_initialized_with_velocity():
    """Momentum SGD should initialize velocity buffers"""
    model, inp, d1, r, d2 = build_test_model()
    model.train()

    opt = SGD(model, lr=0.01, momentum=0.9)
    buffer = opt.get_buffer()

    for module_id, module_buffer in buffer.items():
        for param_name, buf in module_buffer.items():
            assert "velocity" in buf
            assert isinstance(buf["velocity"], np.ndarray)


def test_sgd_gradients_not_modified():
    """SGD should not modify gradient arrays in-place"""
    model, inp, d1, r, d2 = build_test_model()
    model.train()

    opt = SGD(model, lr=0.1, momentum=0.9)

    for module in model.modules:
        if module.has_params:
            for k in module.params:
                module.grads[k] = np.ones_like(module.params[k])

    grad_copies = {}
    for module in model.modules:
        if module.has_params:
            grad_copies[module] = {
                k: v.copy() for k, v in module.grads.items()
            }

    opt.step()

    for module in model.modules:
        if module.has_params:
            for k, v in module.grads.items():
                assert np.allclose(v, grad_copies[module][k])