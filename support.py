import numpy as np


def min_radius(max_angle, num_steps):
    _ceil = np.ceil(num_steps / 2)
    _floor = np.floor(num_steps / 2)
    _min_radius = np.sqrt(
        _ceil**2 + _floor**2 + 2 * _ceil * _floor * np.cos(2 * max_angle)
    )
    return _min_radius


def convert_t_to_phi(t, k: int, max_angle: float, num_steps: int):
    assert 0 <= k <= num_steps - 1
    return max_angle * (2 * (num_steps * t - k) - 1)


def get_k_param(t, num_steps: int):
    return np.ceil(num_steps * t - 1)


def get_phi_param(t, max_angle: float, num_steps: int):
    k = get_k_param(t, num_steps)
    _slope = -2 * max_angle * num_steps
    return _slope * (t - k / num_steps) + max_angle


def support_inner_param(t, max_angle: float, num_steps: int):
    k = get_k_param(t, num_steps)
    phi = get_phi_param(t, max_angle, num_steps)
    _offset = (num_steps - 1 - k) * np.exp(1j * max_angle) + k * np.exp(-1j * max_angle)
    _inner = _offset + np.exp(1j * phi)
    return _inner


def support_outer_param(t, max_angle: float, num_steps: int):
    phi = 2 * max_angle * (t - 0.5)
    return num_steps * np.exp(1j * phi)
