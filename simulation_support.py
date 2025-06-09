import logging

import numpy as np
import matplotlib.pyplot as plt

from support import get_k_param, support_inner_param, support_outer_param, min_radius
from util import export_results

LOGGER = logging.getLogger(__name__)


def main(
    max_angle: float,
    num_steps: int,
    num_samples: int,
    plot: bool = False,
    export: bool = False,
):
    LOGGER.info(f"Starting support simulation with {num_steps:d} steps...")
    t = np.linspace(0, 1, num_samples)
    inner_support = support_inner_param(t, max_angle, num_steps)
    outer_support = support_outer_param(t, max_angle, num_steps)
    _min_radius = min_radius(max_angle, num_steps)
    LOGGER.info(f"Minimum radius: Rmin={_min_radius:.3f}")

    if plot:
        LOGGER.info("Plotting...")
        fig, axs = plt.subplots(1, 2)
        axs[0].set_aspect("equal")
        axs[1].set_aspect("equal")
        for _support in (inner_support, outer_support):
            axs[0].plot(np.real(_support), np.imag(_support), "o-")
            axs[1].plot(np.angle(_support), np.abs(_support), "o-")
            axs[1].hlines(_min_radius, -max_angle, max_angle, ls="--", color="gray")

    if export:
        LOGGER.info("Exporting results...")
        results = {
            "xInner": np.real(inner_support),
            "yInner": np.imag(inner_support),
            "xOuter": np.real(outer_support),
            "yOuter": np.imag(outer_support),
            "radInner": np.abs(inner_support),
            "angleInner": np.angle(inner_support),
            "radOuter": np.abs(outer_support),
            "angleOuter": np.angle(outer_support),
        }
        fname = f"results-support-a{max_angle:.3f}-n{num_steps:n}.dat"
        export_results(results, fname)

    LOGGER.info("Finished all simulations and calculations.")
    return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--max_angle", type=float, required=True)
    parser.add_argument("-N", "--num_steps", type=int, default=10)
    parser.add_argument("--num_samples", type=int, default=1000)
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
