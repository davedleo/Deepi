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

def test_adagrad_initializes_buffers():
    """Adagrad initializes square_sum, lr, and t buffers correctly"""
    model = build_test_model()
    model.train()

    opt = Adagrad(model, lr=0.1, square_sum_init=0.5)

    for module_id, module_buffer in opt.buffer.items():
        module = opt.modules[module_id]
        for k, buf in module_buffer.items():
            assert "square_sum" in buf
            assert "lr" in buf
            assert "t" in buf
            assert np.allclose(buf["square_sum"], 0.5 * np.ones_like(module.params[k]))
            assert np.allclose(buf["lr"], 0.1)
            assert buf["t"] == 0


def test_adagrad_single_step_update():
    """Adagrad updates parameters correctly for first step with new direction method"""
    model = build_test_model()
    model.train()

    opt = Adagrad(model, lr=0.1, square_sum_init=0.0, eps=1e-8)

    for module in model.topology:
        if module.has_params:
            for k in module.params:
                module.grads[k] = np.ones_like(module.params[k])

    orig_params = model.get_params()
    opt.step()

    for module_id, module in opt.modules.items():
        for k, w in module.params.items():
            buf = opt.buffer[module_id][k]
            expected_update = 0.1 * np.ones_like(w) / (np.sqrt(1.0) + 1e-8)
            expected = orig_params[module_id][k] - expected_update
            assert np.allclose(w, expected, atol=1e-7)
            assert buf["t"] == 1


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

    opt.step()
    opt.step()

    for module_id, module in opt.modules.items():
        for k, w in module.params.items():
            buf = opt.buffer[module_id][k]
            # After two steps, square_sum = 2, first update = 0.1/1, second update = 0.1/√2
            total_update = 0.1 / (np.sqrt(1.0) + 1e-8) + 0.1 / (np.sqrt(2.0) + 1e-8)
            expected = orig_params[module_id][k] - total_update * np.ones_like(w)
            assert np.allclose(w, expected, atol=1e-7)
            assert buf["t"] == 2


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

    for module_id, module in opt.modules.items():
        for k, w in module.params.items():
            expected_update = 0.1 * np.ones_like(w) / (np.sqrt(1.0) + 1e-10)
            expected = orig_params[module_id][k] + expected_update
            assert np.allclose(w, expected, atol=1e-7)


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