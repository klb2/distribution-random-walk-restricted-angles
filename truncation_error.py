import logging

import numpy as np
import matplotlib.pyplot as plt

from support import min_radius
from many_steps import cdf_radius_n_large
from util import export_results

LOGGER = logging.getLogger(__name__)


def main(num_steps: int, max_angle: float, plot: bool = False, export: bool = False):
    LOGGER.info("Computing truncation errors of the radius distribution")
    LOGGER.info(f"a={max_angle:.3f}")
    error = []
    steps = np.arange(2, num_steps + 1)
    for n in steps:
        rmin = min_radius(max_angle, n)
        cdf = cdf_radius_n_large(np.array([rmin, n]), max_angle, n)
        _error = 1 - np.diff(cdf)[0]
        LOGGER.info(f"N={n:d}:\t{_error:E}")
        error.append(_error)

    if plot:
        LOGGER.debug("Plotting results...")
        plt.plot(steps, error, ".-")
        plt.xlabel("Number of steps $N$")
        plt.ylabel("Truncation Error/Probability")

    if export:
        results = {"steps": steps, "error": error}
        fname = f"truncation_prob-a{max_angle:.2f}.dat"
        export_results(results, fname)
    return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-N", "--num_steps", type=int, default=2)
    parser.add_argument("-a", "--max_angle", type=float, default=1)
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
