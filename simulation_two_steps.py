import logging
import argparse

import numpy as np
import matplotlib.pyplot as plt

from two_steps import cdf_radius_n2, pdf_angle_n2, pdf_joint_radius_angle_n2
from util import export_results

LOGGER = logging.getLogger(__name__)


def main(max_angle: float, num_samples: int, plot: bool = False, export: bool = False):
    num_bins = 50
    num_plot_points = 250

    phases = (2 * np.random.rand(num_samples, 2) - 1) * max_angle
    result_vector = np.sum(np.exp(1j * phases), axis=1)
    result_radius = np.abs(result_vector)
    result_phases = np.angle(result_vector)

    min_radius = 2 * np.cos(max_angle)
    line_radius = np.linspace(min_radius, 2, num_plot_points)
    line_angle = np.linspace(-max_angle, max_angle, num_plot_points)

    cdf_radius = cdf_radius_n2(line_radius, max_angle)
    pdf_angle = pdf_angle_n2(line_angle, max_angle)
    hist_radius, bins_radius = np.histogram(result_radius, bins=num_bins, density=True)
    cdf_hist_radius = np.cumsum(hist_radius) * (bins_radius[1] - bins_radius[0])
    hist_angle, bins_angle = np.histogram(result_phases, bins=num_bins, density=True)

    if plot:
        fig, axs = plt.subplots()
        axs.hist(result_radius, bins=num_bins, density=True, cumulative=True)
        axs.plot(line_radius, cdf_radius)
        axs.set_xlabel("Radius $R_2$")
        axs.set_ylabel("CDF")

        fig, axs = plt.subplots()
        axs.hist(result_phases, bins=num_bins, density=True, cumulative=False)
        axs.plot(line_angle, pdf_angle)
        axs.set_xlabel("Angle $\\theta_2$")
        axs.set_ylabel("PDF")

    if export:
        results_exact = {
            "radius": line_radius,
            "cdfRad": cdf_radius,
            "angle": line_angle,
            "pdfAngle": pdf_angle,
        }
        fname_exact = f"results-exact-n2-a{max_angle:.3f}.dat"
        export_results(results_exact, fname_exact)

        results_mc = {
            "radius": bins_radius,
            "histRad": np.concat(([0], cdf_hist_radius)),
            "angle": bins_angle,
            "histAngle": np.concat(([0], hist_angle)),
        }
        fname_mc = f"results-mc-n2-a{max_angle:.3f}.dat"
        export_results(results_mc, fname_mc)
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
