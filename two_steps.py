"""
Implementation of the distribution of the two-dimensional random walk with two
steps and restricted angles.
"""

import numpy as np


def pdf_angle_n2(theta, max_angle):
    return np.clip(
        1 / (max_angle) * (1 - np.abs(theta) / max_angle), 0, 1 / (max_angle)
    )


def cdf_angle_n2(theta, max_angle):
    return np.clip(
        (max_angle**2 + 2 * max_angle * theta - np.sign(theta) * theta**2)
        / (2 * max_angle**2),
        0,
        1,
    )


def cdf_radius_n2(radius, max_angle):
    _x = np.arccos(radius**2 / 2 - 1) / 2
    _cdf = 1 - 2 * _x / max_angle + (_x / max_angle) ** 2
    _cdf = (max_angle - np.arccos(radius / 2)) ** 2 / max_angle**2
    return np.clip(_cdf, 0, 1)


def pdf_radius_n2(radius, max_angle):
    _pdf = (
        2
        * (max_angle - np.arccos(radius / 2))
        / (max_angle**2 * np.sqrt(4 - radius**2))
    )
    pdf = np.where(np.logical_and(2 * np.cos(max_angle) < radius, radius < 2), _pdf, 0)
    return pdf


def cdf_radius_giv_angle_n2(radius, theta, max_angle):
    _cdf = 1 - (np.arccos(radius**2 / 2 - 1) / 2) / (max_angle - theta)
    return np.clip(_cdf, 0, 1)


def pdf_joint_radius_angle_n2(radius, theta, max_angle):
    _pdf = radius / (max_angle**2 * np.sqrt(0j+radius**2 * (4 - radius**2)))
    pdf = np.where(
        np.logical_and(
            np.logical_and(2 * np.cos(max_angle - np.abs(theta)) < radius, radius < 2),
            np.logical_and(-max_angle <= theta, theta <= max_angle),
        ),
        _pdf,
        0,
    )
    pdf = np.real_if_close(pdf)
    return pdf
