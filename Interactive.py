import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
    # On the Distribution of a Two-Dimensional Random Walk with Restricted Angles

    _Author:_ Karl-Ludwig Besser (Linköping University)


    This notebook is part of the publication "On the Distribution of a Two-Dimensional Random Walk with Restricted Angles".
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Parameters

    In the following, you find sliders which allow you to adjust the following parameters of the simulations:

    - Maximum angle $a$
    - Number of steps $N$ (in the [section on the approximation for large $N$](#large-number-of-steps))
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Two Steps

    In the following, we will illustrate results for the two-step case, i.e., $N=2$
    """
    )
    return


@app.cell
def _(slider_max_angle):
    slider_max_angle
    return


@app.cell
def _(mo):
    mo.md(r"""### Radius $R_2$""")
    return


@app.cell
def _(cdf_radius_n2, line_radius_2, max_angle, mo, np, plt, result_radius_2):
    _fig, _axs = plt.subplots()
    _axs.set_ylim([0, 1])
    _axs.set_xlim([2 * np.cos(max_angle), 2])

    _cdf_rad = cdf_radius_n2(line_radius_2, max_angle=max_angle)
    _axs.plot(line_radius_2, _cdf_rad)
    _axs.hist(
        result_radius_2, bins=50, density=True, cumulative=True, histtype="step"
    )
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(mo):
    mo.md(r"""### Angle $\theta_2$""")
    return


@app.cell
def _(line_phase, max_angle, mo, pdf_angle_n2, plt, result_phases_2):
    _fig, _axs = plt.subplots()
    _axs.set_xlim([-max_angle, max_angle])

    _pdf_ang = pdf_angle_n2(line_phase, max_angle=max_angle)
    _axs.plot(line_phase, _pdf_ang)
    _axs.hist(
        result_phases_2, bins=50, density=True, cumulative=False, histtype="step"
    )
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(mo):
    mo.md(r"""### Joint Distribution of $(R_2, \theta_2)$""")
    return


@app.cell
def _(
    line_phase,
    line_radius_2,
    max_angle,
    mo,
    np,
    pdf_joint_radius_angle_n2,
    plt,
    result_phases_2,
    result_radius_2,
):
    _fig, _axs = plt.subplots(1, 2, squeeze=True)
    for _ax in _axs:
        _ax.set_xlim([-max_angle, max_angle])
        _ax.set_ylim([2 * np.cos(max_angle), 2])

    _R, _T = np.meshgrid(line_radius_2, line_phase)

    _pdf_joint = pdf_joint_radius_angle_n2(_R, _T, max_angle=max_angle)
    _axs[0].pcolormesh(_T, _R, _pdf_joint)
    _axs[1].hist2d(result_phases_2, result_radius_2, bins=(50, 40), density=True)
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Support

    In the following, we illustrate the support of the resulting vector after $N$ steps.
    """
    )
    return


@app.cell
def _(slider_max_angle):
    slider_max_angle
    return


@app.cell
def _(slider_num_components):
    slider_num_components
    return


@app.cell
def _(mo):
    mo.md(r"""### Cartesian Coordinates""")
    return


@app.cell
def _(
    line_phase,
    max_angle,
    mo,
    np,
    num_components,
    num_plot_points,
    plt,
    result_vector,
):
    _phi = np.linspace(-np.pi, np.pi, num_plot_points)
    _fig, _ax = plt.subplots(1, 2, squeeze=True)
    _axs = _ax[0]
    _axs.set_aspect("equal", "box")

    _axs.set_xlim([0, num_components])
    _axs.set_ylim([-num_components, num_components])

    _axs.plot(
        num_components * np.cos(_phi),
        num_components * np.sin(_phi),
        "r--",
    )
    _axs.plot(
        [
            num_components * np.cos(max_angle),
            0,
            num_components * np.cos(max_angle),
        ],
        [
            num_components * np.sin(max_angle),
            0,
            -num_components * np.sin(max_angle),
        ],
        "r--",
    )

    _axs.plot(
        [0, num_components * np.cos(np.pi / 4)],
        [0, num_components * np.sin(np.pi / 4)],
        "g--",
    )

    _axs.vlines(
        num_components * np.cos(max_angle),
        num_components * np.sin(-max_angle),
        num_components * np.sin(max_angle),
        ls="--",
        color="orange",
    )

    _axs.plot(
        [0, num_components],
        [
            -num_components * np.cos(max_angle) / np.tan(max_angle)
            + (num_components - 2) * np.sin(max_angle),
            num_components / np.tan(max_angle)
            - num_components * np.cos(max_angle) / np.tan(max_angle)
            + (num_components - 2) * np.sin(max_angle),
        ],
        ls="--",
        color="gray",
    )

    for _n in range(num_components):
        _center = _n * np.exp(1j * max_angle) + (num_components - 1 - _n) * np.exp(
            -1j * max_angle
        )
        _angle = np.arctan2(
            np.sin(max_angle) * (2 * _n + 2 - num_components),
            np.cos(max_angle) * num_components,
        )
        _axs.plot(
            [0, num_components * np.cos(_angle)],
            [0, num_components * np.sin(_angle)],
            "k-.",
        )
        for __axs in _ax:
            __axs.plot(
                np.real(_center + np.exp(1j * line_phase)),
                np.imag(_center + np.exp(1j * line_phase)),
                "b--",
            )


    _ax[1].set_aspect("equal", "box")
    _ax[1].scatter(np.real(result_vector), np.imag(result_vector), s=2)


    mo.mpl.interactive(_fig)
    return


@app.cell
def _(mo):
    mo.md(r"""### Polar Coordinates""")
    return


@app.cell
def _(max_angle, min_radius, mo, np, num_components, plt, support):
    _fig, _axs = plt.subplots(1, 2, squeeze=True)
    _t = np.linspace(0, 1, 1000)
    _inner_support = support.support_inner_param(_t, max_angle, num_components)
    _outer_support = support.support_outer_param(_t, max_angle, num_components)
    _axs[0].set_aspect("equal")
    _axs[1].set_aspect("equal")
    for _support in (_inner_support, _outer_support):
        _axs[0].plot(np.real(_support), np.imag(_support), "-")
        _axs[1].plot(np.angle(_support), np.abs(_support), "-")
        _axs[1].hlines(min_radius, -max_angle, max_angle, ls="--", color="gray")

    mo.mpl.interactive(_fig)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Large Number of Steps

    In the following, we will take a look a the approximation for a large number of steps $N$.
    """
    )
    return


@app.cell
def _(slider_max_angle):
    slider_max_angle
    return


@app.cell
def _(slider_num_components):
    slider_num_components
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Radius $R_N$

    First, we show interactive plots on the distribution of the radius $R_N$ after $N$ steps.
    The plots show both the approximation for large $N$ and a histogram obtained through Monte Carlo simulations.
    """
    )
    return


@app.cell
def _(
    cdf_radius_n_large,
    line_radius,
    max_angle,
    min_radius,
    mo,
    num_components,
    plt,
    result_radius,
):
    _fig, _axs = plt.subplots()
    _axs.set_ylim([0, 1])
    _axs.set_xlim([min_radius, num_components])

    _cdf_rad = cdf_radius_n_large(
        line_radius, max_angle=max_angle, num_steps=num_components
    )
    _axs.plot(line_radius, _cdf_rad)
    _axs.hist(
        result_radius, bins=50, density=True, cumulative=True, histtype="step"
    )
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Angle $\theta_N$

    Next, we show interactive plots on the distribution of the angle $\theta_N$ after $N$ steps.
    The plots show both the approximation for large $N$ and a histogram obtained through Monte Carlo simulations.
    """
    )
    return


@app.cell
def _(
    line_phase,
    max_angle,
    mo,
    num_components,
    pdf_angle_n_large,
    plt,
    result_phases,
):
    _fig, _axs = plt.subplots()
    _axs.set_xlim([-max_angle, max_angle])

    _pdf_ang = pdf_angle_n_large(
        line_phase, max_angle=max_angle, num_steps=num_components
    )
    _axs.plot(line_phase, _pdf_ang)
    _axs.hist(
        result_phases, bins=50, density=True, cumulative=False, histtype="step"
    )
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Code

    In the following, we have necessary imports and code.
    As Marimo builds a dependency tree of the cells, they can be in any order.
    Therefore, we can keep the messy parts at the end of the notebook.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Parameter Definitions

    In this section, we define (global) parameters and interactive sliders.
    """
    )
    return


@app.cell
def _(mo, np):
    slider_max_angle = mo.ui.slider(
        0.1, 0.5 * np.pi, 0.02, label="Maximum angle $a$"
    )
    slider_num_components = mo.ui.slider(4, 20, 1, label="Number of steps $N$")
    return slider_max_angle, slider_num_components


@app.cell
def _(slider_max_angle, slider_num_components):
    max_angle = slider_max_angle.value
    num_components = slider_num_components.value
    return max_angle, num_components


@app.cell
def _(max_angle, np, num_components, support):
    num_samples = 100000
    num_plot_points = 250

    min_radius = support.min_radius(max_angle, num_components)

    phases = (2 * np.random.rand(num_samples, num_components) - 1) * max_angle
    result_vector = np.sum(np.exp(1j * phases), axis=1)
    result_radius = np.abs(result_vector)
    result_phases = np.angle(result_vector)
    line_radius = np.linspace(min_radius, num_components, num_plot_points)
    line_phase = np.linspace(-max_angle, max_angle, num_plot_points)
    return (
        line_phase,
        line_radius,
        min_radius,
        num_plot_points,
        phases,
        result_phases,
        result_radius,
        result_vector,
    )


@app.cell
def _(max_angle, np, num_plot_points, phases):
    line_radius_2 = np.linspace(2 * np.cos(max_angle), 2, num_plot_points)
    result_vector_2 = np.sum(np.exp(1j * phases[:, :2]), axis=1)
    result_radius_2 = np.abs(result_vector_2)
    result_phases_2 = np.angle(result_vector_2)
    return line_radius_2, result_phases_2, result_radius_2


@app.cell
def _(mo):
    mo.md(r"""### Function Definitions""")
    return


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(r"""### Imports""")
    return


@app.cell
def _():
    import marimo as mo

    import os

    os.environ["PYTHONWARNINGS"] = "ignore"

    import numpy as np
    import matplotlib.pyplot as plt

    from two_steps import cdf_radius_n2, pdf_angle_n2, pdf_joint_radius_angle_n2
    import support
    from many_steps import (
        cdf_radius_n_large,
        pdf_angle_n_large,
        pdf_joint_radius_angle_n_large,
    )
    return (
        cdf_radius_n2,
        cdf_radius_n_large,
        mo,
        np,
        pdf_angle_n2,
        pdf_angle_n_large,
        pdf_joint_radius_angle_n2,
        plt,
        support,
    )


if __name__ == "__main__":
    app.run()
