#! bin/bash

# This script assumes that both conda and git are already installed.
# Also, make sure to install libalglib as:
# sudo apt install libalglib-dev
# Tested on:
# * * * Ubuntu-22.04 (via WSL2) * * * 

# Before executing the script run:
# conda create --name g6x  #(if g6x is nonexistent)
# conda activate g6x

conda install update 
conda install install autotools-dev autoconf libtoolize
conda install install g++

conda install -y install libgmp10
conda install install libmpfr-dev 
conda install libqd-dev

conda env list | grep g6x
LATTICE_ENV_CHECK=$?
if [ $LATTICE_ENV_CHECK -ne 0 ]; then
        echo "Need to create g6x conda environment."
        exit 1
fi

conda install fpylll cython cysignals flake8 ipython numpy begins pytest requests scipy multiprocessing-logging matplotlib autoconf automake libtoo
git clone "https://github.com/Summwer/cvp-g6k-cpu-solver.git"

cd ./cvp-g6k-cpu-solver
echo "- - -"
echo "$PWD"
echo "- - -"
git reset --hard "442ae40"

cp ../patch.patch ./
cp ../benchmark_slicer_pump.py ./
cp ../lattice_reduction.py ./
cp ../global_consts.py ./
cp ../utils.py ./
cp ../sample.py ./
cp ../discretegauss.py ./

# apt-get download libalglib-dev
# dpkg -x libalglib-dev_*.deb ./libalglib

git apply ./patch.patch




conda install setuptools

git clone https://github.com/cr-marcstevens/parallel-hashmap

# pip install virtualenv
# PYTHON=python3 ./bootstrap.sh
# source ./activate

make clean
./configure CXX=/usr/bin/g++
python setup.py build_ext --inplace