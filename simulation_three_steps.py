import logging
import argparse

import numpy as np
import matplotlib.pyplot as plt

from three_steps import cdf_radius_n3, pdf_angle_n3
from many_steps import pdf_angle_n_large, cdf_radius_n_large
from util import export_results


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
    num_steps = 3
    num_bins = 50
    num_plot_points = 20

    phases = (2 * np.random.rand(num_samples, num_steps) - 1) * max_angle
    result_vector = np.sum(np.exp(1j * phases), axis=1)
    result_radius = np.abs(result_vector)
    result_phases = np.angle(result_vector)

    min_radius = np.sqrt(0.5 * (1 + 9 + 8 * np.cos(2 * max_angle)))
    line_radius = np.linspace(min_radius, num_steps, num_plot_points)
    line_angle = np.linspace(-max_angle, max_angle, num_plot_points)

    pdf_angle = pdf_angle_n3(line_angle, max_angle)
    cdf_radius = cdf_radius_n3(line_radius, max_angle)

    pdf_angle_approx = pdf_angle_n_large(line_angle, max_angle, num_steps)
    cdf_radius_approx = cdf_radius_n_large(line_radius, max_angle, num_steps)

    angles_support = []
    radius_support = []
    for k in range(num_steps):
        _angles = np.linspace(
            np.arctan(
                ((num_steps - 2 - 2 * k) * np.sin(max_angle))
                / (num_steps * np.cos(max_angle))
            ),
            np.arctan(
                ((num_steps - 2 * k) * np.sin(max_angle))
                / (num_steps * np.cos(max_angle))
            ),
            100,
        )
        print(f"k={k:d}, angles=[{min(_angles)}, {max(_angles)}]")
        _radius_support = inner_radius_approx(_angles, max_angle, num_steps, k=k)
        angles_support.extend(_angles)
        radius_support.extend(_radius_support)
    _idx_sort = np.argsort(angles_support)
    angles_support = np.array(angles_support)[_idx_sort]
    radius_support = np.array(radius_support)[_idx_sort]

    hist_joint, bins_angle_joint, bins_radius_joint = np.histogram2d(
        result_phases, result_radius, bins=num_bins, density=True
    )
    hist_joint = hist_joint.T
    mesh_angle, mesh_radius = np.meshgrid(bins_angle_joint[:-1], bins_radius_joint[:-1])

    hist_radius, bins_radius = np.histogram(result_radius, bins=num_bins, density=True)
    cdf_hist_radius = np.cumsum(hist_radius) * (bins_radius[1] - bins_radius[0])
    hist_angle, bins_angle = np.histogram(result_phases, bins=num_bins, density=True)

    if plot:
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

        fig, axs = plt.subplots()
        plt.hist2d(result_phases, result_radius, bins=num_bins, density=True)
        plt.plot(
            angles_support,
            radius_support,
            "r-.",
        )

    if export:
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
        fname_support = f"results-n3-support-a{max_angle:.3f}.dat"
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
            "radius": np.ravel(mesh_radius),
            "angle": np.ravel(mesh_angle),
            "pdfJoint": np.ravel(hist_joint),
        }
        fname_joint = f"results-joint-n3-a{max_angle:.3f}.dat"
        export_results(results_joint, fname_joint)
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
