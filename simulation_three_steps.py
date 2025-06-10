import logging
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from three_steps import cdf_radius_n3, pdf_angle_n3, pdf_joint_radius_angle_n3
from many_steps import pdf_angle_n_large, cdf_radius_n_large
from support import min_radius, support_inner_param
from util import export_results


os.environ["PYTHONWARNINGS"] = "ignore"

LOGGER = logging.getLogger(__name__)


def inner_radius_approx(theta, max_angle, num_steps: int, k: int):
    phi = _approx_phase_inner_radius(theta, max_angle, num_steps, k)
    # phi = np.linspace(-max_angle, max_angle, len(theta))
    _part1 = (num_steps - 1 - k) ** 2 + k**2 + 1
    _part2 = (
        (num_steps - 1 - k) * k * np.cos(2 * max_angle)
        + (num_steps - 1 - k) * np.cos(max_angle - phi)
        + k * np.cos(max_angle + phi)
    )
    radius = np.sqrt(_part1 + 2 * _part2)
    return radius


def _approx_phase_inner_radius(theta, max_angle: float, num_steps: int, k: int):
    _m = (1 + (num_steps - 1) * np.cos(max_angle)) * (
        1
        + (((num_steps - 2 * k - 1) * np.sin(max_angle)) ** 2)
        / (1 + (num_steps - 1) * np.cos(max_angle)) ** 2
    )
    # _m = 1.0 / _m

    _offset = np.arctan(
        ((num_steps - 2 * k - 1) * np.sin(max_angle))
        / (1 + (-1 + num_steps) * np.cos(max_angle))
    )
    return _m * (theta - _offset)


def main(max_angle: float, num_samples: int, plot: bool = False, export: bool = False):
    LOGGER.info("Starting simulation with three steps...")
    num_steps = 3
    num_bins = 50
    num_plot_points = 250

    min_rad = min_radius(max_angle, num_steps)
    line_radius = np.linspace(min_rad, num_steps, num_plot_points)
    line_angle = np.linspace(-max_angle, max_angle, num_plot_points)

    _support_inner = support_inner_param(np.linspace(0, 1, 100), max_angle, num_steps)
    angles_support = np.angle(_support_inner)
    radius_support = np.abs(_support_inner)
    LOGGER.debug("Finished preparations.")

    LOGGER.info("Calculating marginal distributions...")
    pdf_angle = pdf_angle_n3(line_angle, max_angle)
    LOGGER.debug("Finished calculating angle distribution.")
    cdf_radius = cdf_radius_n3(line_radius, max_angle)
    LOGGER.debug("Finished calculating radius distribution.")
    LOGGER.info("Finished calculating marginal distributions.")

    LOGGER.info("Calculating large N approximations...")
    pdf_angle_approx = pdf_angle_n_large(line_angle, max_angle, num_steps)
    cdf_radius_approx = cdf_radius_n_large(line_radius, max_angle, num_steps)
    LOGGER.info("Finished large N approximations.")

    LOGGER.info("Calculating joint distribution...")
    _R, _T = np.meshgrid(
        np.linspace(min_rad, num_steps*(1+1/num_bins), num_bins),
        np.linspace(-max_angle, max_angle, num_bins),
    )
    # _R, _T = np.meshgrid(line_radius, line_angle)
    pdf_joint = pdf_joint_radius_angle_n3(_R, _T, max_angle)
    LOGGER.info("Finished calculating joint distribution.")

    LOGGER.info("Starting Monte Carlo simulation...")
    phases = (2 * np.random.rand(num_samples, num_steps) - 1) * max_angle
    result_vector = np.sum(np.exp(1j * phases), axis=1)
    result_radius = np.abs(result_vector)
    result_phases = np.angle(result_vector)
    hist_joint, bins_angle_joint, bins_radius_joint = np.histogram2d(
        result_phases, result_radius, bins=num_bins, density=True
    )
    hist_joint = hist_joint.T
    mesh_angle, mesh_radius = np.meshgrid(bins_angle_joint[:-1], bins_radius_joint[:-1])

    hist_radius, bins_radius = np.histogram(result_radius, bins=num_bins, density=True)
    cdf_hist_radius = np.cumsum(hist_radius) * (bins_radius[1] - bins_radius[0])
    hist_angle, bins_angle = np.histogram(result_phases, bins=num_bins, density=True)
    LOGGER.info("Finished Monte Carlo simulations.")

    if plot:
        LOGGER.info("Plotting...")
        fig, axs = plt.subplots()
        axs.hist(result_radius, bins=num_bins, density=True, cumulative=True)
        axs.plot(line_radius, cdf_radius, label="Numerical Integration")
        axs.plot(line_radius, cdf_radius_approx, label="Large $N$ Approximation")
        axs.set_xlabel("Radius $R_3$")
        axs.set_ylabel("CDF")
        fig.legend()

        fig, axs = plt.subplots()
        axs.hist(result_phases, bins=num_bins, density=True, cumulative=False)
        axs.plot(line_angle, pdf_angle, label="Numerical Integration")
        axs.plot(line_angle, pdf_angle_approx, label="Large $N$ Approximation")
        axs.set_xlabel("Angle $\\theta_2$")
        axs.set_ylabel("PDF")
        fig.legend()

        fig, axs = plt.subplots(2)
        axs[0].hist2d(result_phases, result_radius, bins=num_bins, density=True)
        axs[0].plot(
            angles_support,
            radius_support,
            "r-.",
        )
        axs[0].set_xlim([-max_angle, max_angle])
        axs[0].set_ylim([min_rad, num_steps])
        axs[0].set_title("Histogram")
        axs[1].pcolormesh(_T, _R, pdf_joint)
        axs[1].set_xlim([-max_angle, max_angle])
        axs[1].set_ylim([min_rad, num_steps])
        axs[1].set_title("Calculation")
        fig.suptitle(f"Joint PDF of Radius and Angle for $a={max_angle:.3f}$ and $N=3$")

    if export:
        LOGGER.info("Exporting results...")
        results = {
            "radius": line_radius,
            "angle": line_angle,
            "pdfAngle": pdf_angle,
            "cdfRadius": cdf_radius,
            "pdfAngleApprox": pdf_angle_approx,
            "cdfRadiusApprox": cdf_radius_approx,
        }
        fname = f"results-n3-a{max_angle:.3f}.dat"
        export_results(results, fname)

        results_support = {
            "radius": radius_support,
            "angle": angles_support,
        }
        fname_support = f"results-support-n3-a{max_angle:.3f}.dat"
        export_results(results_support, fname_support)

        results_mc = {
            "radius": bins_radius,
            "histRad": np.concat(([0], cdf_hist_radius)),
            "angle": bins_angle,
            "histAngle": np.concat(([0], hist_angle)),
        }
        fname_mc = f"results-mc-n3-a{max_angle:.3f}.dat"
        export_results(results_mc, fname_mc)

        results_joint = {
            # "radius": np.ravel(mesh_radius),
            # "angle": np.ravel(mesh_angle),
            # "pdfJoint": np.ravel(hist_joint),
            "radius": np.ravel(_R),
            "angle": np.ravel(_T),
            "pdfJoint": np.ravel(pdf_joint),
        }
        fname_joint = f"results-joint-n3-a{max_angle:.3f}.dat"
        export_results(results_joint, fname_joint)

    LOGGER.info("Finished all simulations and calculations.")
    return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--max_angle", type=float, required=True)
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
