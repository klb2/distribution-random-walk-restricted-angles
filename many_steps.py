"""
Implementation of the approximate distribution of the two-dimensional random
walk with a large number of steps and restricted angles.
"""

import numpy as np
from scipy import stats

import gx2


def _mean_x(max_angle):
    return np.sin(max_angle) / max_angle


def _var_x(max_angle):
    part1 = (max_angle + np.cos(max_angle) * np.sin(max_angle)) / (2 * max_angle)
    part2 = _mean_x(max_angle) ** 2
    return part1 - part2


def _var_y(max_angle):
    return (max_angle - np.cos(max_angle) * np.sin(max_angle)) / (2 * max_angle)


def _cdf_ratio_n_large(w, max_angle, num_steps):
    _mx = _mean_x(max_angle)
    _vx = _var_x(max_angle)  # /num_steps
    _vy = _var_y(max_angle)  # /num_steps
    _aw = np.sqrt(w**2 / _vx + 1 / _vy)
    _arg = _mx * w / (np.sqrt(_vx * _vy) * _aw)
    _cdf = stats.norm.cdf(_arg)
    return _cdf


def _pdf_ratio_n_large(w, max_angle, num_steps):
    _mx = _mean_x(max_angle)
    _vx = _var_x(max_angle)
    _vy = _var_y(max_angle)
    _arg = (_mx * num_steps * w) / (np.sqrt(num_steps * (w**2 * _vx + _vy)))
    _deriv = _mx * np.sqrt(num_steps) * _vy / (w**2 * _vx + _vy) ** (3 / 2)
    _pdf = stats.norm.pdf(_arg) * _deriv
    return _pdf


def pdf_angle_n_large(theta, max_angle: float, num_steps: int):
    _part1 = _pdf_ratio_n_large(np.tan(theta), max_angle, num_steps)
    _part2 = 1 / np.cos(theta) ** 2
    return _part1 * _part2


def cdf_angle_n_large(theta, max_angle: float, num_steps: int):
    return _cdf_ratio_n_large(np.tan(theta), max_angle, num_steps)


def pdf_radius_n_large(radius, max_angle: float, num_steps: int):
    _mx = _mean_x(max_angle)
    _vx = _var_x(max_angle)  # /num_steps
    _vy = _var_y(max_angle)  # /num_steps
    _k = [1, 1]
    # _w = [_vx / num_steps, _vy / num_steps]
    # _lambda = [_mx**2 * num_steps / _vx, 0]
    _w = [_vx * num_steps, _vy * num_steps]
    _lambda = [_mx**2 * num_steps / _vx, 0]
    # _pdf_approx = gx2.gx2_pdf_ruben(radius**2, _w, _k, _lambda, m=0)
    _pdf_approx = gx2.gx2_pdf_imhof(radius**2, _w, _k, _lambda, s=0, m=0)
    _pdf_approx = _pdf_approx * 2 * radius
    return _pdf_approx


def cdf_radius_n_large(radius, max_angle: float, num_steps: int):
    _mx = _mean_x(max_angle)
    _vx = _var_x(max_angle)  # /num_steps
    _vy = _var_y(max_angle)  # /num_steps
    _k = [1, 1]
    _w = [_vx * num_steps, _vy * num_steps]
    _lambda = [_mx**2 * num_steps / _vx, 0]
    _cdf = gx2.gx2_cdf_imhof(radius**2, _w, _k, _lambda, s=0, m=0)
    _cdf = np.clip(_cdf, 0, 1)
    return _cdf


def pdf_joint_radius_angle_n_large(radius, theta, max_angle: float, num_steps: int):
    _mx = _mean_x(max_angle)
    _vx = _var_x(max_angle)
    _vy = _var_y(max_angle)
    _stdx = np.sqrt(_vx)
    _stdy = np.sqrt(_vy)
    _factor = radius / (num_steps**2 * _stdx * _stdy)
    _part1 = stats.norm.pdf(
        (radius * np.cos(theta) - num_steps * _mx) / (num_steps * _stdx)
    )
    _part2 = stats.norm.pdf((radius * np.sin(theta)) / (num_steps * _stdy))
    _pdf = _factor * _part1 * _part2
    return _pdf
