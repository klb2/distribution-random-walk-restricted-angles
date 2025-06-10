#!/bin/sh

# ...
# Information about the paper...
# ...
#
# Copyright (C) 20XX ...
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
