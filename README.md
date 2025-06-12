# On the Distribution of a Two-Dimensional Random Walk with Restricted Angles

[![Marimo](https://img.shields.io/badge/Launch-Marimo_notebook-hsl(168%2C61%25%2C28%25))](https://marimo.app/?src=https%3A%2F%2Fraw.githubusercontent.com%2Fklb2%2Fdistribution-random-walk-restricted-angles%2Frefs%2Fheads%2Fmaster%2FInteractive.py)
![GitHub](https://img.shields.io/github/license/klb2/distribution-random-walk-restricted-angles)


This repository is accompanying the paper "On the Distribution of a Two-Dimensional Random Walk with Restricted Angles" (Karl-Ludwig Besser, Jun. 2025).

The idea is to give an interactive version of the calculations and presented
concepts to the reader. One can also change different parameters and explore
different behaviors on their own.


## File List
The following files are provided in this repository:

- `run.sh`: Bash script that reproduces the figures presented in the paper.
- `util.py`: Python module that contains utility functions, e.g., for saving results.
- `simulation_two_steps.py`: Python script that contains the simulation for a random walk with two steps.
- `simulation_three_steps.py`: Python script that contains the simulation for a random walk with three steps.
- `simulation_many_steps.py`: Python script that contains the simulation for a random walk with many steps (large N approximation).
- `simulation_support.py`: Python script that contains the simulation for exploring the support of the random walk.
- `gx2.py`: Python module that contains functions to calculate the PDF and CDF of a generalized chi-square distribution.
- `many_steps.py`: Python module that contains the functions for the large N approximation.
- `single_step.py`: Python module that contains the functions for a random walk with a single step.
- `two_steps.py`: Python module that contains the functions for a random walk with two steps.
- `three_steps.py`: Python module that contains the functions for a random walk with three steps.
- `support.py`: Python module that contains the functions to calculate the support of the joint distribution

## Usage
### Running it online
The easiest way is to use the official [marimo](https://marimo.app/) playground
to run the notebook online. Simply navigate to [https://marimo.app/?src=https%3A%2F%2Fraw.githubusercontent.com%2Fklb2%2Fdistribution-random-walk-restricted-angles%2Frefs%2Fheads%2Fmaster%2FInteractive.py](https://marimo.app/?src=https%3A%2F%2Fraw.githubusercontent.com%2Fklb2%2Fdistribution-random-walk-restricted-angles%2Frefs%2Fheads%2Fmaster%2FInteractive.py)
to run the notebooks in your browser without setting everything up locally.

### Local Installation
If you want to run it locally on your machine, Python3 and marimo (for the
interactive notebook) are needed.
The present code was developed and tested with the following versions:

- Python 3.13.3
- numpy 2.3.0
- scipy 1.15.3
- pandas 2.2.3
- joblib 1.4.2

Make sure you have [Python3](https://www.python.org/downloads/) installed on
your computer.
You can then install the required packages by running
```bash
pip3 install -r requirements.txt
```
This will install all the needed packages which are listed in the requirements 
file.


Finally, you can run the Marimo notebooks with
```bash
marimo run Interactive.py
```

You can also recreate the figures from the paper by running
```bash
bash run.sh
```


## Acknowledgements
This research was supported by Security Link.


## License and Referencing
This program is licensed under the MIT license. If you in any way use this
code for research that results in publications, please cite our original
article listed above.

You can use the following BibTeX entry
```bibtex
@article{Besser2025distribution,
  author = {Besser, Karl-Ludwig},
  title = {On the Distribution of a Two-Dimensional Random Walk with Restricted Angles},
  ...
}
