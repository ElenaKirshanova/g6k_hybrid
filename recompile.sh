#!/usr/bin/env bash


make clean

# add --with-stats-slicer=1 to enable statistic collection for slicer
# add --with-stats=1 or --with-stats=3 to enable statistic collection for siever

./configure CXX=/usr/bin/g++
python setup.py build_ext --inplace