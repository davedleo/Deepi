import numpy as np
import pytest

from deepi import Model
from deepi.modules import Input, Dense, ReLU
from deepi.optimization.optimizers import RAdam


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

def test_radam_initializes_buffers():
    """RAdam initializes velocity, square_avg, t, and rho_inf"""
    model = build_test_model()
    model.train()

    opt = RAdam(model)

    for module_buffer in opt.buffer.values():
        for buf in module_buffer.values():
            assert "t" in buf
            assert "velocity" in buf
            assert "square_avg" in buf
            assert "rho_inf" in buf

            assert isinstance(buf["velocity"], np.ndarray)
            assert isinstance(buf["square_avg"], np.ndarray)
            assert isinstance(buf["rho_inf"], float)
            assert buf["t"] == 0


def test_radam_single_step_no_rectification():
    """
    First RAdam step:
    rho_t <= 5, so rectification is NOT applied
    """
    model = build_test_model()
    model.train()

    lr = 0.001
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    opt = RAdam(
        model,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
    )

    # set gradients = 1 everywhere
    for module in model.topology:
        if module.has_params:
            for k in module.params:
                module.grads[k] = np.ones_like(module.params[k])

    orig_params = model.get_params()
    opt.step()

    for name, module in model.modules.items():
        if not module.has_params:
            continue

        for k, w in module.params.items():
            dw = np.ones_like(w)

            # ---- step t = 1 ----
            t = 1

            # ---- velocity ----
            v = (1.0 - beta1) * dw
            v_hat = v / (1.0 - beta1)

            # ---- square average ----
            s = (1.0 - beta2) * (dw ** 2)

            # ---- rho_t ----
            rho_inf = 2.0 / (1.0 - beta2) - 1.0
            rho_t = rho_inf - 2.0 * t * (beta2 ** t) / (1.0 - beta2 ** t)

            # rho_t <= 5 → no rectification
            assert rho_t <= 5.0

            expected = orig_params[name][k] - lr * v_hat
            assert np.allclose(w, expected)


def test_radam_two_steps_with_rectification():
    """
    After enough steps, rectification may or may not occur within two steps depending on beta2
    """
    model = build_test_model()
    model.train()

    lr = 0.01
    beta1 = 0.9
    beta2 = 0.9     # smaller beta2 → rectification earlier
    eps = 1e-8

    opt = RAdam(
        model,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
    )

    for module in model.topology:
        if module.has_params:
            for k in module.params:
                module.grads[k] = np.ones_like(module.params[k])

    orig_params = model.get_params()

    # ---- step 1 ----
    opt.step()
    # ---- step 2 ----
    opt.step()

    for name, module in model.modules.items():
        if not module.has_params:
            continue

        for k, w in module.params.items():
            dw = np.ones_like(w)

            # =========================
            # Step 1
            # =========================
            t1 = 1
            v1 = (1.0 - beta1) * dw
            s1 = (1.0 - beta2) * (dw ** 2)

            v_hat1 = v1 / (1.0 - beta1)

            rho_inf = 2.0 / (1.0 - beta2) - 1.0
            rho_t1 = rho_inf - 2.0 * t1 * (beta2 ** t1) / (1.0 - beta2 ** t1)

            if rho_t1 > 5.0:
                l1 = np.sqrt(1.0 - beta2 ** t1) / (np.sqrt(s1) + eps)
                r1 = np.sqrt(
                    (rho_t1 ** 2 - 6 * rho_t1 + 8) * rho_inf
                ) / np.sqrt(
                    (rho_inf ** 2 - 6 * rho_inf + 8) * rho_t1
                )
                update1 = lr * v_hat1 * r1 * l1
            else:
                update1 = lr * v_hat1

            # =========================
            # Step 2
            # =========================
            t2 = 2
            v2 = beta1 * v1 + (1.0 - beta1) * dw
            s2 = beta2 * s1 + (1.0 - beta2) * (dw ** 2)

            v_hat2 = v2 / (1.0 - beta1 ** t2)

            rho_t2 = rho_inf - 2.0 * t2 * (beta2 ** t2) / (1.0 - beta2 ** t2)

            if rho_t2 > 5.0:
                l2 = np.sqrt(1.0 - beta2 ** t2) / (np.sqrt(s2) + eps)
                r2 = np.sqrt(
                    (rho_t2 ** 2 - 6 * rho_t2 + 8) * rho_inf
                ) / np.sqrt(
                    (rho_inf ** 2 - 6 * rho_inf + 8) * rho_t2
                )
                update2 = lr * v_hat2 * r2 * l2
            else:
                update2 = lr * v_hat2

            expected = orig_params[name][k] - (update1 + update2)
            assert np.allclose(w, expected)


def test_radam_maximize_flag():
    """maximize=True inverts update direction"""
    model = build_test_model()
    model.train()

    opt = RAdam(model, maximize=True)

    for module in model.topology:
        if module.has_params:
            for k in module.params:
                module.grads[k] = np.ones_like(module.params[k])

    orig_params = model.get_params()
    opt.step()

    for name, module in model.modules.items():
        if module.has_params:
            for k, w in module.params.items():
                # parameters should move in opposite direction
                assert np.any(w > orig_params[name][k])


def test_radam_does_not_modify_gradients():
    """RAdam must not modify gradients in-place"""
    model = build_test_model()
    model.train()

    opt = RAdam(model)

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