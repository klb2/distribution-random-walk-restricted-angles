import logging
import argparse

import numpy as np
import matplotlib.pyplot as plt

from many_steps import (
    pdf_angle_n_large,
    cdf_radius_n_large,
    pdf_radius_n_large,
    pdf_joint_radius_angle_n_large,
)
from util import export_results

LOGGER = logging.getLogger(__name__)


def main(
    max_angle: float,
    num_steps: int,
    num_samples: int,
    plot: bool = False,
    export: bool = False,
):
    num_bins = 50
    num_plot_points = 250

    phases = (2 * np.random.rand(num_samples, num_steps) - 1) * max_angle
    result_vector = np.sum(np.exp(1j * phases), axis=1)
    result_radius = np.abs(result_vector)
    result_phases = np.angle(result_vector)

    _ceil = np.ceil(num_steps / 2)
    _floor = np.floor(num_steps / 2)
    min_radius = np.sqrt(
        _ceil**2 + _floor**2 + 2 * _ceil * _floor * np.cos(2 * max_angle)
    )
    line_radius = np.linspace(min_radius, num_steps, num_plot_points)
    line_angle = np.linspace(-max_angle, max_angle, num_plot_points)

    # pdf_radius = pdf_radius_n_large(line_radius, max_angle, num_steps)
    cdf_radius = cdf_radius_n_large(line_radius, max_angle, num_steps)
    pdf_angle = pdf_angle_n_large(line_angle, max_angle, num_steps)

    hist_joint, bins_angle_joint, bins_radius_joint = np.histogram2d(
        result_phases, result_radius, bins=num_bins, density=True
    )
    hist_joint = hist_joint.T
    mesh_angle, mesh_radius = np.meshgrid(bins_angle_joint, bins_radius_joint)

    _R, _T = np.meshgrid(line_radius[:-1:5], line_angle[:-1:5])
    # _R, _T = np.meshgrid(line_radius, line_angle)
    pdf_joint = pdf_joint_radius_angle_n_large(_R, _T, max_angle, num_steps)

    if plot:
        fig, axs = plt.subplots()
        axs.hist(
            result_radius,
            bins=num_bins,
            density=True,
            cumulative=True,
            label="Histogram",
        )
        axs.plot(line_radius, cdf_radius, label="Approximation")
        axs.legend()
        axs.set_xlabel("Radius $R_N$")
        axs.set_ylabel("CDF")
        axs.set_title(
            f"CDF of the Radius for $a={max_angle:.3f}$ and $N={num_steps:d}$"
        )

        fig, axs = plt.subplots()
        axs.hist(
            result_phases,
            bins=num_bins,
            density=True,
            cumulative=False,
            label="Histogram",
        )
        axs.plot(line_angle, pdf_angle, label="Approximation")
        axs.legend()
        axs.set_xlabel("Angle $\\theta_N$")
        axs.set_ylabel("PDF")
        axs.set_title(f"PDF of the Angle for $a={max_angle:.3f}$ and $N={num_steps:d}$")

        fig, axs = plt.subplots(2)
        _plot = axs[0].pcolormesh(mesh_radius, mesh_angle, hist_joint)
        # axs[0].hist2d(result_radius, result_phases, bins=num_bins, density=True)
        axs[0].plot(2 * np.cos(max_angle - np.abs(line_angle)), line_angle, "r--")
        axs[0].set_xlim([min_radius, num_steps])
        axs[0].set_ylim([-max_angle, max_angle])
        axs[0].set_title("Histogram")
        fig.colorbar(_plot)
        _plot = axs[1].pcolormesh(_R, _T, pdf_joint)  # * num_plot_points / num_bins)
        # axs[1].plot(2 * np.cos(max_angle - np.abs(line_angle)), line_angle, "r--")
        axs[1].set_xlim([min_radius, num_steps])
        axs[1].set_ylim([-max_angle, max_angle])
        axs[1].set_title("Approximation")
        fig.suptitle(
            f"Joint PDF of Radius and Angle for $a={max_angle:.3f}$ and $N={num_steps:d}$"
        )
        fig.colorbar(_plot)

    if export:
        hist_radius, bins_radius = np.histogram(
            result_radius, bins=num_bins, density=True
        )
        cdf_hist_radius = np.cumsum(hist_radius) * (bins_radius[1] - bins_radius[0])
        hist_angle, bins_angle = np.histogram(
            result_phases, bins=num_bins, density=True
        )

        results_approx = {
            "radius": line_radius,
            "cdfRad": cdf_radius,
            "angle": line_angle,
            "pdfAngle": pdf_angle,
        }
        fname_approx = f"results-approx-n-large-a{max_angle:.3f}-n{num_steps:n}.dat"
        export_results(results_approx, fname_approx)

        results_mc = {
            "radius": bins_radius,
            "histRad": np.concat(([0], cdf_hist_radius)),
            "angle": bins_angle,
            "histAngle": np.concat(([0], hist_angle)),
        }
        fname_mc = f"results-mc-n-large-a{max_angle:.3f}-n{num_steps:n}.dat"
        export_results(results_mc, fname_mc)

        results_joint = {
            "radius": np.ravel(_R),
            "angle": np.ravel(_T),
            "pdfJoint": np.ravel(pdf_joint),
        }
        fname_joint = f"results-joint-n-large-a{max_angle:.3f}-n{num_steps:n}.dat"
        export_results(results_joint, fname_joint)
    return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--max_angle", type=float, required=True)
    parser.add_argument("-N", "--num_steps", type=int, default=10)
    parser.add_argument("--num_samples", type=int, default=100_000)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--export", action="store_true")
    parser.add_argument(
        "-v", "--verbosity", action="count", default=0, help="Increase output verbosity"
    )
    args = vars(parser.parse_args())
    verb = args.pop("verbosity")
    logging.basicConfig(
        format="%(asctime)s - [%(levelname)8s]: %(message)s",
        handlers=[
            logging.FileHandler("main.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    loglevel = logging.WARNING - verb * 10
    LOGGER.setLevel(loglevel)
    main(**args)
    plt.show()
