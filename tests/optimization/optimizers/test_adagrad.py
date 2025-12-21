import numpy as np
import pytest

from deepi import Model
from deepi.modules import Input, Dense, ReLU
from deepi.optimization.optimizers import Adagrad


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

def test_adagrad_initializes_square_sum():
    """Adagrad initializes square_sum buffers correctly"""
    model = build_test_model()
    model.train()

    opt = Adagrad(model, lr=0.1, square_sum_init=0.5)

    for module_buffer in opt.buffer["params"].values():
        for buf in module_buffer.values():
            assert "square_sum" in buf
            assert np.allclose(buf["square_sum"], 0.5 * np.ones_like(buf["square_sum"]))


def test_adagrad_single_step_update():
    """Adagrad updates parameters correctly for first step"""
    model = build_test_model()
    model.train()

    opt = Adagrad(model, lr=0.1, square_sum_init=0.0, eps=1e-8)

    for module in model.topology:
        if module.has_params:
            for k in module.params:
                module.grads[k] = np.ones_like(module.params[k])

    orig_params = model.get_params()
    opt.step()

    # First step: square_sum = 0 + 1^2 = 1 → update = lr * dw / sqrt(square_sum + eps) ≈ 0.1
    for name, module in model.modules.items():
        if module.has_params:
            for k, w in module.params.items():
                expected = orig_params[name][k] - 0.1 * np.ones_like(w)
                assert np.allclose(w, expected, atol=1e-7)


def test_adagrad_multiple_steps_accumulate_square_sum():
    """Adagrad accumulates square_sum across multiple steps"""
    model = build_test_model()
    model.train()

    opt = Adagrad(model, lr=0.1, square_sum_init=0.0, eps=1e-8)

    for module in model.topology:
        if module.has_params:
            for k in module.params:
                module.grads[k] = np.ones_like(module.params[k])

    orig_params = model.get_params()

    # Step 1: square_sum = 1 → update1 = 0.1
    opt.step()
    # Step 2: square_sum = 1 + 1 = 2 → update2 = 0.1 / sqrt(2) ≈ 0.0707107
    opt.step()

    for name, module in model.modules.items():
        if module.has_params:
            for k, w in module.params.items():
                total_update = 0.1 + 0.1 / np.sqrt(2)
                expected = orig_params[name][k] - total_update * np.ones_like(w)
                assert np.allclose(w, expected, atol=1e-7)


def test_adagrad_maximize_flag():
    """maximize=True inverts gradient direction"""
    model = build_test_model()
    model.train()

    opt = Adagrad(model, lr=0.1, maximize=True)

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


def test_adagrad_does_not_modify_gradients():
    """Adagrad must not modify gradients in-place"""
    model = build_test_model()
    model.train()

    opt = Adagrad(model, lr=0.1)

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