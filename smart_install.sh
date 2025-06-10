#! bin/bash

# This script assumes that both conda and git are already installed.
# Tested on:
# * * * Ubuntu-22.04 (via WSL2) * * * 

# Before executing the script run:
# conda create --name g6x  #(if g6x is nonexistent)
# conda activate g6x

# sudo apt-get update 
# sudo apt-get install build-essential autotools-dev autoconf g++

# sudo apt-get -y install libgmp10
# sudo apt-get install libmpfr-dev 
# sudo apt install libqd-dev

conda env list | grep g6x
LATTICE_ENV_CHECK=$?
if [ $LATTICE_ENV_CHECK -ne 0 ]; then
        echo "Need to create g6x conda environment."
        exit 1
fi

conda install build-essential autoconf g++


conda install fpylll cython cysignals flake8 ipython numpy begins pytest requests scipy multiprocessing-logging matplotlib autoconf automake libtool
git clone "https://github.com/Summwer/cvp-g6k-cpu-solver.git"

cd ./cvp-g6k-cpu-solver
git reset --hard "442ae40"

cp ../patch.patch ./
cp ../benchmark_slicer_pump.py ./
cp ../lattice_reduction.py ./
cp ../global_consts.py ./
cp ../utils.py ./
cp ../sample.py ./
cp ../discretegauss.py ./

git apply ./patch.patch

# sudo apt install libalglib-dev # should already be installed
conda install setuptools

git clone https://github.com/cr-marcstevens/parallel-hashmap

make clean
./configure CXX=/usr/bin/g++
python setup.py build_ext --inplace