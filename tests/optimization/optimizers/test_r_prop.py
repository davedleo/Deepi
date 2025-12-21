import numpy as np
import pytest

from deepi import Model
from deepi.modules import Input, Dense, ReLU
from deepi.optimization.optimizers import Rprop


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

def test_rprop_initializes_buffers():
    """Rprop should initialize previous gradient and eta buffers"""
    model = build_test_model()
    model.train()

    opt = Rprop(model, lr=0.01)

    for module_buffer in opt.buffer["params"].values():
        for buf in module_buffer.values():
            assert "dw_prev" in buf
            assert "eta" in buf
            assert np.all(buf["dw_prev"] == 0)
            assert np.all(buf["eta"] == 0.01)


def test_rprop_single_step_positive_gradients():
    """Rprop should increase steps on positive gradient direction"""
    model = build_test_model()
    model.train()

    opt = Rprop(model, lr=0.1, eta_plus=1.2, eta_minus=0.5, min_step=1e-6, max_step=1.0)

    # Set all gradients to positive ones
    for module in model.topology:
        if module.has_params:
            for k in module.params:
                module.grads[k] = np.ones_like(module.params[k])

    orig_params = model.get_params()
    opt.step()

    for name, module in model.modules.items():
        if module.has_params:
            for k, w in module.params.items():
                eta = opt.buffer["params"][name][k]["eta"]
                expected_step = np.sign(module.grads[k]) * eta
                expected_params = orig_params[name][k] - expected_step
                assert np.allclose(w, expected_params)


def test_rprop_step_size_adjustment():
    """Rprop eta increases/decreases correctly based on gradient signs"""
    model = build_test_model()
    model.train()

    opt = Rprop(model, lr=0.1, eta_plus=2.0, eta_minus=0.5, min_step=1e-6, max_step=1.0)

    # Initial positive gradient → eta should increase
    for module in model.topology:
        if module.has_params:
            for k in module.params:
                module.grads[k] = np.ones_like(module.params[k])

    opt.step()

    # Next step: same positive gradient → eta multiplied by eta_plus
    for module in model.topology:
        if module.has_params:
            for k in module.params:
                module.grads[k] = np.ones_like(module.params[k])

    prev_eta = {name: {k: buf["eta"].copy() for k, buf in module_buffer.items()}
                for name, module_buffer in opt.buffer["params"].items()}

    opt.step()

    for name, module in model.modules.items():
        if module.has_params:
            for k, buf in opt.buffer["params"][name].items():
                assert np.all(buf["eta"] <= opt.max_step)
                assert np.all(buf["eta"] >= prev_eta[name][k])


def test_rprop_negative_gradient_decreases_eta():
    """Rprop decreases eta on negative gradient product"""
    model = build_test_model()
    model.train()

    opt = Rprop(model, lr=0.1, eta_plus=1.5, eta_minus=0.5, min_step=1e-6, max_step=1.0)

    # Step 1: positive gradient
    for module in model.topology:
        if module.has_params:
            for k in module.params:
                module.grads[k] = np.ones_like(module.params[k])
    opt.step()

    # Step 2: negative gradient (flip sign) → eta decreases
    for module in model.topology:
        if module.has_params:
            for k in module.params:
                module.grads[k] = -np.ones_like(module.params[k])

    prev_eta = {name: {k: buf["eta"].copy() for k, buf in module_buffer.items()}
                for name, module_buffer in opt.buffer["params"].items()}

    opt.step()

    for name, module_buffer in opt.buffer["params"].items():
        for k, buf in module_buffer.items():
            assert np.all(buf["eta"] <= prev_eta[name][k])
            assert np.all(buf["eta"] >= opt.min_step)


def test_rprop_maximize_flag():
    """maximize=True inverts update direction"""
    model = build_test_model()
    model.train()

    opt = Rprop(model, lr=0.1, maximize=True)

    # All positive gradients
    for module in model.topology:
        if module.has_params:
            for k in module.params:
                module.grads[k] = np.ones_like(module.params[k])

    orig_params = model.get_params()
    opt.step()

    for name, module in model.modules.items():
        if module.has_params:
            for k, w in module.params.items():
                eta = opt.buffer["params"][name][k]["eta"]
                expected_step = np.sign(module.grads[k]) * eta  
                expected_params = orig_params[name][k] + expected_step
                assert np.allclose(w, expected_params)


def test_rprop_does_not_modify_gradients():
    """Rprop must not modify gradients in-place"""
    model = build_test_model()
    model.train()

    opt = Rprop(model, lr=0.1)

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