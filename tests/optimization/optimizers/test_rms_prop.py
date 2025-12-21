import numpy as np
import pytest

from deepi import Model
from deepi.modules import Input, Dense, ReLU
from deepi.optimization.optimizers import RMSprop


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

def test_rmsprop_basic_update():
    """RMSprop without momentum should normalize gradients by RMS"""
    model, inp, d1, r, d2 = build_test_model()
    model.train()

    lr = 0.1
    alpha = 0.9
    eps = 1e-8
    opt = RMSprop(model, lr=lr, alpha=alpha, centered=False)

    for module in model.topology:
        if module.has_params:
            for k in module.params:
                module.grads[k] = np.ones_like(module.params[k])

    orig_params = model.get_params()
    opt.step()

    # square_avg = (1 - alpha) * 1^2
    square_avg = (1 - alpha)
    expected_update = lr / (np.sqrt(square_avg) + eps)

    for name, module in model.modules.items():
        if module.has_params:
            for k, v in module.params.items():
                assert np.allclose(
                    v,
                    orig_params[name][k] - expected_update * np.ones_like(v)
                )


def test_rmsprop_two_steps_accumulates_square_avg():
    """square_avg should accumulate across steps"""
    model, inp, d1, r, d2 = build_test_model()
    model.train()

    lr = 0.1
    alpha = 0.5
    eps = 1e-8
    opt = RMSprop(model, lr=lr, alpha=alpha, centered=False)

    for module in model.topology:
        if module.has_params:
            for k in module.params:
                module.grads[k] = np.ones_like(module.params[k])

    orig_params = model.get_params()

    # step 1
    opt.step()
    square_avg_1 = (1 - alpha)

    # step 2
    opt.step()
    square_avg_2 = alpha * square_avg_1 + (1 - alpha)
    
    for name, module in model.modules.items():
        if module.has_params:
            for k, v in module.params.items():
                expected = (
                    orig_params[name][k]
                    - lr / (np.sqrt(square_avg_1) + eps)
                    - lr / (np.sqrt(square_avg_2) + eps)
                )
                assert np.allclose(v, expected)


def test_rmsprop_with_momentum_single_step():
    """Momentum RMSprop should store velocity on first step"""
    model, inp, d1, r, d2 = build_test_model()
    model.train()

    lr = 0.1
    mu = 0.9
    alpha = 0.9
    eps = 1e-8
    opt = RMSprop(
        model,
        lr=lr,
        momentum=mu,
        alpha=alpha,
        centered=False,
    )

    for module in model.topology:
        if module.has_params:
            for k in module.params:
                module.grads[k] = np.ones_like(module.params[k])

    orig_params = model.get_params()
    opt.step()

    square_avg = (1 - alpha)
    v = 1.0 / (np.sqrt(square_avg) + eps)
    expected_update = lr * v

    for name, module in model.modules.items():
        if module.has_params:
            for k, v_param in module.params.items():
                assert np.allclose(
                    v_param,
                    orig_params[name][k] - expected_update * np.ones_like(v_param)
                )


def test_rmsprop_with_momentum_two_steps():
    """Velocity should accumulate across steps"""
    model, inp, d1, r, d2 = build_test_model()
    model.train()

    lr = 0.1
    mu = 0.5
    alpha = 0.5
    eps = 1e-8
    opt = RMSprop(
        model,
        lr=lr,
        momentum=mu,
        alpha=alpha,
        centered=False,
    )

    for module in model.topology:
        if module.has_params:
            for k in module.params:
                module.grads[k] = np.ones_like(module.params[k])

    orig_params = model.get_params()

    # step 1
    opt.step()
    square_avg_1 = (1 - alpha)
    v1 = 1.0 / (np.sqrt(square_avg_1) + eps)

    # step 2
    opt.step()
    square_avg_2 = alpha * square_avg_1 + (1 - alpha)
    v2 = mu * v1 + 1.0 / (np.sqrt(square_avg_2) + eps)

    for name, module in model.modules.items():
        if module.has_params:
            for k, v in module.params.items():
                expected = orig_params[name][k] - lr * (v1 + v2)
                assert np.allclose(v, expected)


def test_rmsprop_centered():
    """Centered RMSprop should subtract squared mean"""
    model, inp, d1, r, d2 = build_test_model()
    model.train()

    lr = 0.1
    alpha = 0.9
    eps = 1e-8
    opt = RMSprop(
        model,
        lr=lr,
        alpha=alpha,
        centered=True,
    )

    for module in model.topology:
        if module.has_params:
            for k in module.params:
                module.grads[k] = np.ones_like(module.params[k])

    orig_params = model.get_params()
    opt.step()

    square_avg = (1 - alpha)
    avg = (1 - alpha)
    centered_var = square_avg - avg ** 2
    expected_update = lr / (np.sqrt(centered_var) + eps)

    for name, module in model.modules.items():
        if module.has_params:
            for k, v in module.params.items():
                assert np.allclose(
                    v,
                    orig_params[name][k] - expected_update * np.ones_like(v)
                )


def test_rmsprop_maximize_flag():
    """maximize=True should invert gradient direction"""
    model, inp, d1, r, d2 = build_test_model()
    model.train()

    opt = RMSprop(model, lr=0.1, maximize=True, centered=False)

    for module in model.topology:
        if module.has_params:
            for k in module.params:
                module.grads[k] = np.ones_like(module.params[k])

    orig_params = model.get_params()
    opt.step()

    for name, module in model.modules.items():
        if module.has_params:
            for k, v in module.params.items():
                assert np.allclose(
                    v,
                    orig_params[name][k] + np.abs(v - orig_params[name][k])
                )


def test_rmsprop_buffer_initialized():
    """RMSprop should initialize square_avg (and velocity if needed)"""
    model, inp, d1, r, d2 = build_test_model()
    model.train()

    opt = RMSprop(model, lr=0.01, momentum=0.9, centered=True)
    buffer = opt.get_buffer()

    for module_id, module_buffer in buffer.items():
        for param_name, buf in module_buffer.items():
            assert "square_avg" in buf
            assert isinstance(buf["square_avg"], np.ndarray)
            assert "avg" in buf
            assert isinstance(buf["avg"], np.ndarray)
            assert "velocity" in buf
            assert isinstance(buf["velocity"], np.ndarray)


def test_rmsprop_gradients_not_modified():
    """RMSprop should not modify gradient arrays in-place"""
    model, inp, d1, r, d2 = build_test_model()
    model.train()

    opt = RMSprop(model, lr=0.1, momentum=0.9)

    for module in model.topology:
        if module.has_params:
            for k in module.params:
                module.grads[k] = np.ones_like(module.params[k])

    grad_copies = {}
    for module in model.topology:
        if module.has_params:
            grad_copies[module] = {
                k: v.copy() for k, v in module.grads.items()
            }

    opt.step()

    for module in model.topology:
        if module.has_params:
            for k, v in module.grads.items():
                assert np.allclose(v, grad_copies[module][k])