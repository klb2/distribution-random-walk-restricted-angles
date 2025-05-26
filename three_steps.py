""" """

import numpy as np
from scipy import integrate
import joblib
import matplotlib.pyplot as plt

from single_step import cdf_angle_n1, pdf_angle_n1
from two_steps import pdf_joint_radius_angle_n2, cdf_radius_n2, cdf_angle_n2


def cdf_radius_n3(radius, max_angle: float):
    radius = np.array(radius)

    _integrand = lambda r, p, x: (
        cdf_angle_n1(
            p + np.arccos(np.clip((x**2 - r**2 - 1) / (2 * r), -1, 1)), max_angle
        )
        - cdf_angle_n1(
            p - np.arccos(np.clip((x**2 - r**2 - 1) / (2 * r), -1, 1)), max_angle
        )
    ) * pdf_joint_radius_angle_n2(r, p, max_angle)

    _lower_bound_radius = lambda p, x: (
        2 * np.cos(max_angle - np.abs(p)),
        2 - np.finfo(float).eps,
    )

    # integral = lambda x: integrate.dblquad(
    #    _integrand, -max_angle, max_angle, _lower_bound_radius, 2, args=(x,)
    # )[0]
    integral = lambda x: integrate.nquad(
        _integrand,
        ranges=[_lower_bound_radius, [-max_angle, max_angle]],
        args=(x,),
        opts={"limit": 100, "epsabs": 1e-5},
    )[0]

    _cdf = joblib.Parallel(n_jobs=int(0.85 * joblib.cpu_count()))(
        joblib.delayed(integral)(_t) for _t in radius
    )
    cdf = 1.0 - np.array(_cdf)
    print(cdf)
    return cdf


def cdf_angle_n3(theta, max_angle: float):
    theta = np.array(theta)

    _integrand = lambda r, p, t: cdf_angle_n1(
        (1 + r) * t - r * p, max_angle
    ) * pdf_joint_radius_angle_n2(r, p, max_angle)

    _lower_bound_radius = lambda p: 2 * np.cos(max_angle - np.abs(p))

    integral = lambda x: integrate.dblquad(
        _integrand, -max_angle, max_angle, _lower_bound_radius, 2, args=(x,)
    )[0]

    _cdf = joblib.Parallel(n_jobs=int(0.85 * joblib.cpu_count()))(
        joblib.delayed(integral)(_t) for _t in theta
    )
    cdf = np.array(_cdf)
    # result = integrate.dblquad(
    #    _integrand, 0, max_angle, _lower_bound_radius, 2, args=(theta,)
    # )
    return cdf


def pdf_angle_n3(theta, max_angle: float):
    theta = np.array(theta)

    _integrand = (
        lambda r, p, t: pdf_angle_n1((1 + r) * t - r * p, max_angle)
        * pdf_joint_radius_angle_n2(r, p, max_angle)
        * (1 + r)
    )
    _bounds_radius = lambda p, t: (
        2 * np.cos(max_angle - np.abs(p)),
        2 - np.finfo(float).eps,
    )

    integral = lambda x: integrate.nquad(
        _integrand,
        [_bounds_radius, [-max_angle, max_angle]],
        args=(x,),
        opts={"limit": 100},
    )[0]

    _pdf = joblib.Parallel(n_jobs=int(0.85 * joblib.cpu_count()))(
        joblib.delayed(integral)(_t) for _t in theta
    )
    pdf = np.array(_pdf)
    # result = integrate.dblquad(
    #    _integrand, 0, max_angle, _lower_bound_radius, 2, args=(theta,)
    # )
    return pdf


def main():
    max_angle = 0.5
    # t = np.linspace(-max_angle, max_angle, 10)
    # r = cdf_angle_n3(t, max_angle)
    # r = pdf_angle_n3(t, max_angle)
    r = np.linspace(2.7, 3, 10)
    result = cdf_radius_n3(r, max_angle)
    plt.plot(r, result)
    return


if __name__ == "__main__":
    main()
    plt.show()
