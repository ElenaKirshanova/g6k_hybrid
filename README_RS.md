******************************
The Randomized Slicer in the General Sieve Kernel (G6K) library
******************************

The Randomized Slicer is a C++ and Python extension of the `G6K library<https://github.com/fplll/g6k>` that implements the batch-CVP algorithm from `Doulgerakis-Laarhoven-de Weger "Finding closest
lattice vectors using approximate Voronoi cells."<https://eprint.iacr.org/2016/888.pdf>`

The code is based on BDGL implementation from `Ducas-Stevens-van Woerden "Advanced lattice  sieving on GPUs, with tensor cores"<https://eprint.iacr.org/2021/141.pdf>`

Building the library
====================

You will the `G6K library <https://github.com/fplll/g6k>`. Building on Lunix usually works by running ``bootstrap.sh`` (see comprehensive instruction at the `G6K repository <https://github.com/fplll/g6k>`):

.. code-block:: bash

    # once only: creates local python env, builds fplll, fpylll and G6K
    ./bootstrap.sh [ -j # ]
    
    # for every new shell: activates local python env
    source ./activate                   

On systems with co-existing python2 and 3, you can force a specific version installation using ``PYTHON=<pythoncmd> ./boostrap.sh`` instead.
The number of parallel compilation jobs can be controlled with `-j #`.


## Potential Solution to solve issues building on ARM-Macs (see `Issue <https://github.com/fplll/g6k/issues/128>`)



Running RandomizedSlicer
====================
In order to test-run our randomized slicer, run the script test_slicer.py.

.. code-block:: bash 
    python test_slicer.py TODO




Reproducing the experiments from the paper
====================



# Reproducing Figure 2
...

