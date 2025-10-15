import logging

import numpy as np
import matplotlib.pyplot as plt

from support import support_inner_param
from many_steps import pdf_joint_radius_angle_n_large
from two_steps import pdf_joint_radius_angle_n2
from util import export_results

LOGGER = logging.getLogger(__file__)


def _two_steps_likelihood(max_angle: float, num_samples: int):
    LOGGER.info(f"Starting Monte Carlo simulation for max_angle={max_angle:.2f}")
    num_steps = 2
    phases = (2 * np.random.rand(num_samples, num_steps) - 1) * max_angle
    result_vector = np.sum(np.exp(1j * phases), axis=1)
    likelihood = pdf_joint_radius_angle_n2(
        np.abs(result_vector), np.angle(result_vector), max_angle=max_angle
    )
    return likelihood


def main(
    num_samples: int,
    plot: bool = False,
    export: bool = False,
):
    num_steps_large = 30
    max_angle_large = 0.5
    LOGGER.info(f"Starting the over-the-air computation example")
    t = np.linspace(0, 1, 250)
    inner_support_large = support_inner_param(t, max_angle_large, num_steps_large)
    points_large = np.array([[0.1, 28.5], [-0.3, 27]]).T
    likelihoods_large = pdf_joint_radius_angle_n_large(
        *points_large[::-1], max_angle=max_angle_large, num_steps=num_steps_large
    )
    LOGGER.info(f"Testing points: {points_large.T}")
    LOGGER.info(f"Their likelihoods for the points are: {likelihoods_large}")

    # max_angle_two, inner_support_two, point_two, pdf_point_two = _two_steps()
    max_angle = [0.1, 0.5, np.pi / 4]
    # likelihood_threshold = np.linspace(0, 10, 100)
    likelihood_threshold = np.linspace(-1, 10, 150)
    false_alarm_prob = {}
    for _a in max_angle:
        likelihood_two = _two_steps_likelihood(_a, num_samples)
        likelihood_two = np.log(likelihood_two)
        _false_alarm = np.reshape(likelihood_two, (-1, 1)) < likelihood_threshold
        fa_prob = np.mean(_false_alarm, axis=0)
        false_alarm_prob[_a] = fa_prob

    if plot:
        LOGGER.info("Plotting...")
        fig, axs = plt.subplots(1, 2)
        axs[0].plot(np.angle(inner_support_large), np.abs(inner_support_large), "o-")
        axs[0].hlines(num_steps_large, -max_angle_large, max_angle_large, ls="--")
        axs[0].scatter(*points_large, color="r")
        axs[0].set_xlabel("Resulting Angle")
        axs[0].set_ylabel("Resulting Radius")
        axs[0].set_title("Example 10: Detecting Necessary Resynchronization")

        for _a, _prob in false_alarm_prob.items():
            axs[1].plot(likelihood_threshold, _prob, label=f"$a={_a:.2f}$")
        axs[1].legend()
        axs[1].set_xlabel("Log-likelihood threshold $\\gamma$")
        axs[1].set_ylabel("Probability of false alarm $P_{\\text{FA}}$")

    if export:
        LOGGER.info("Exporting results...")
        results = {f"a{k:.2f}": v for k, v in false_alarm_prob.items()}
        results["gamma"] = likelihood_threshold
        fname = f"results-example-ota.dat"
        export_results(results, fname)

    LOGGER.info("Finished all simulations and calculations.")
    return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=1000000)
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
