"""
Implementation of the distribution of the two-dimensional random walk with a
single step and restricted angles.
"""

import numpy as np


def pdf_angle_n1(theta, max_angle):
    return np.where(np.abs(theta) < max_angle, 1 / (2 * max_angle), 0)


def cdf_angle_n1(theta, max_angle):
    return np.clip((theta + max_angle) / (2 * max_angle), 0, 1)
