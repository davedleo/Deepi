import numpy as np
import pytest

from deepi import Model
from deepi.modules import Input, Dense, ReLU
from deepi.optimization.optimizers import Adam


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

def test_adam_initializes_buffers():
    """Adam initializes velocity, square_avg, square_avg_max (if amsgrad) and t buffers correctly"""
    model = build_test_model()
    model.train()

    opt = Adam(model, lr=0.01, amsgrad=True)

    for module_id, module_buffer in opt.buffer.items():
        module = opt.modules[module_id]
        for k, buf in module_buffer.items():
            assert "velocity" in buf
            assert "square_avg" in buf
            assert "t" in buf
            assert "square_avg_max" in buf
            assert np.allclose(buf["velocity"], 0.0)
            assert np.allclose(buf["square_avg"], 0.0)
            assert np.allclose(buf["square_avg_max"], 0.0)
            assert buf["t"] == 0


def test_adam_single_step_update():
    """Adam updates parameters correctly for first step"""
    model = build_test_model()
    model.train()

    opt = Adam(model, lr=0.1, beta1=0.0, beta2=0.0, eps=1e-8)  # simplify: velocity = dw, square_avg = dw^2

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


def test_adam_multiple_steps_accumulate_buffers():
    """Adam accumulates velocity and square_avg across multiple steps"""
    model = build_test_model()
    model.train()

    opt = Adam(model, lr=0.1, beta1=0.5, beta2=0.5, eps=1e-8)

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
            # velocity and square_avg evolve according to beta1, beta2
            assert buf["t"] == 2
            # ensure velocity and square_avg are non-zero
            assert np.all(buf["velocity"] != 0)
            assert np.all(buf["square_avg"] != 0)


def test_adam_amsgrad_behavior():
    """Adam with amsgrad keeps square_avg_max non-decreasing"""
    model = build_test_model()
    model.train()

    opt = Adam(model, lr=0.1, beta1=0.0, beta2=0.0, eps=1e-8, amsgrad=True)

    for module in model.topology:
        if module.has_params:
            for k in module.params:
                module.grads[k] = np.ones_like(module.params[k]) * (np.random.rand(*module.params[k].shape) + 0.5)

    opt.step()
    buf_prev = {module_id: {k: v["square_avg_max"].copy() for k, v in module_buf.items()}
                for module_id, module_buf in opt.buffer.items()}

    opt.step()

    for module_id, module_buf in opt.buffer.items():
        for k, buf in module_buf.items():
            assert np.all(buf["square_avg_max"] >= buf_prev[module_id][k])


def test_adam_maximize_flag():
    """maximize=True inverts gradient direction"""
    model = build_test_model()
    model.train()

    opt = Adam(model, lr=0.1, beta1=0.0, beta2=0.0, maximize=True)

    for module in model.topology:
        if module.has_params:
            for k in module.params:
                module.grads[k] = np.ones_like(module.params[k])

    orig_params = model.get_params()
    opt.step()

    for module_id, module in opt.modules.items():
        for k, w in module.params.items():
            expected_update = 0.1 * np.ones_like(w) / (np.sqrt(1.0) + 1e-8)
            expected = orig_params[module_id][k] + expected_update
            assert np.allclose(w, expected, atol=1e-7)


def test_adam_does_not_modify_gradients():
    """Adam must not modify gradients in-place"""
    model = build_test_model()
    model.train()

    opt = Adam(model, lr=0.1)

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