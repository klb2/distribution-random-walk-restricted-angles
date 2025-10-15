#!/bin/sh

# This repository is accompanying the publication
# "On the Distribution of a Two-Dimensional Random Walk with Restricted Angles"
# (Karl-Ludwig Besser, Jul. 2025. arXiv:2507.15475).
#
# Copyright (C) 2025 Karl-Ludwig Besser
# License: MIT

echo "Figures 2, 4, and 5: Distributions for N=2"
python simulation_two_steps.py -v -a .5 --num_samples 1000000 --plot --export

echo "Support"
python simulation_support.py -v -a 0.85 -N 3 --plot --export
python simulation_support.py -v -a 1.4 -N 4 --plot --export

echo "Three Steps"
python simulation_three_steps.py -vv -a .5 --num_samples=10000000 --plot --export

echo "Approximation for large N"
python simulation_many_steps.py -v -a .5 -N 5 --num_samples 10000000 --plot --export
python simulation_many_steps.py -v -a .5 -N 30 --num_samples 10000000 --plot --export
python simulation_support.py -v -a 0.5 -N 30 --plot --export
python simulation_many_steps.py -v -a 1.25 -N 5 --num_samples 10000000 --plot --export
python simulation_many_steps.py -v -a 1.25 -N 30 --num_samples 10000000 --plot --export

echo "OtA example"
python example_ota_computation.py -v --plot
