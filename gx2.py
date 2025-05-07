"""
Python implementation of the generalized chi-square distribution.

This module provides functions to numerically calculate the PDF of a
generalized chi-square distribution.
This implementation is adapted from the Matlab toolbox "gx2" by Abhranil Das.
(https://github.com/abhranildas/gx2)
"""
import numpy as np
from scipy import stats, special, integrate
import joblib


def _gx2_imhof_integrand_pdf(u, x, w, k, lam, s, m):
    theta = (
        np.sum(k * np.atan(w * u) + (lam * (w * u)) / (1 + w**2 * u**2), axis=0) / 2
        + u * (m - x) / 2
    )
    rho = np.prod(
        ((1 + w**2 * u**2) ** (k / 4))
        * np.exp(((w**2 * u**2) * lam) / (2 * (1 + w**2 * u**2))),
        axis=0,
    ) * np.exp(u**2 * s**2 / 8)
    f = np.cos(theta) / rho
    return f

def gx2_pdf_imhof(x, w, k, lam, s, m):
    w = np.array(w)
    x = np.array(x)
    k = np.array(k)
    lam = np.array(lam)
    _integrand = lambda y: integrate.quad(
       _gx2_imhof_integrand_pdf, 0, np.inf, args=(y, w, k, lam, s, m)
    )[0]
    _pdf = joblib.Parallel(int(joblib.cpu_count() * 0.85))(
       joblib.delayed(_integrand)(_x) for _x in x
    )
    # for _x in x:
    # result = integrate.quad(_gx2_imhof_integrand_pdf, 0, np.inf, args=(_x, w, k, lam, s, m))
    pdf = np.array(_pdf) / (2 * np.pi)
    return pdf


def gx2_pdf_ruben(x, w, k, lam, m: float = 0, num_ruben: int = 100):
    w = np.array(w)
    x = np.array(x)
    k = np.array(k)
    lam = np.array(lam)
    assert np.all(w > 0)
    beta = 0.90625 * np.min(w)
    M = np.sum(k)
    _n = np.reshape(np.arange(1, num_ruben), (-1, 1))
    _g1 = np.sum(k * (1 - beta / w) ** _n, axis=1)
    _g2 = beta * _n * ((1 - beta / w) ** (_n - 1) @ np.reshape(lam / w, (-1, 1)))
    _g = _g1 + np.ravel(_g2)  # num_ruben-1
    a = np.zeros((num_ruben, 1), dtype=np.float128)
    a[0] = np.sqrt(np.exp(-np.sum(lam)) * beta**M * np.prod(w ** (-k)))
    for idx in range(1, num_ruben):
        a[idx] = (np.flip(_g[:idx]) @ a[:idx]) / (2 * idx)
    x_grid, k_grid = np.meshgrid(
        (x - m) / beta, np.arange(M, M + 2 * (num_ruben - 1) + 1, 2)
    )
    _F = stats.chi2(k_grid).pdf(x_grid)
    p = np.ravel(a) @ _F
    p = p / beta
    return p


if __name__ == "__main__":
    x = np.linspace(0, 200, 1000)
    w = np.array([1.0, 3.0, 4.0])
    k = np.array([1, 2, 3])
    lam = np.array([2, 3, 7])
    m = 0
    s = 0
    pdf_ruben = gx2_pdf_ruben(x, w, k, lam, m=m)
    pdf_imhof = gx2_pdf_imhof(x, w, k, lam, s=s, m=m)

    import matplotlib.pyplot as plt

    plt.plot(x, pdf_ruben)
    plt.plot(x, pdf_imhof, '--')
    plt.show()
