******************************
The Randomized Slicer in the General Sieve Kernel (G6K) library
******************************

The Randomized Slicer is a C++ and Python extension of the `G6K library <https://github.com/fplll/g6k>`_ that implements the batch-CVP algorithm from Doulgerakis-Laarhoven-de Weger `"Finding closest
lattice vectors using approximate Voronoi cells" <https://eprint.iacr.org/2016/888.pdf>`_.

The code is based on BDGL implementation from Ducas-Stevens-van Woerden `"Advanced lattice  sieving on GPUs, with tensor cores" <https://eprint.iacr.org/2021/141.pdf>`_

Building the library
====================

You will need the `G6K library <https://github.com/fplll/g6k>`_. Building on Lunix usually works by running ``bootstrap.sh`` (see comprehensive instruction at the `G6K repository <https://github.com/fplll/g6k>`_):

.. code-block:: bash

    # once only: creates local python env, builds fplll, fpylll and G6K
    ./bootstrap.sh [ -j # ]
    
    # for every new shell: activates local python env
    source ./activate                   

On systems with co-existing python2 and 3, you can force a specific version installation using ``PYTHON=<pythoncmd> ./boostrap.sh`` instead.
The number of parallel compilation jobs can be controlled with `-j #`.


Potential Solution to solve issues building on ARM-Macs (see `Issue #128 <https://github.com/fplll/g6k/issues/128>`_)
-----------------------------------------------------------------------------------------------------------------



Running RandomizedSlicer
====================
To test-run our randomized slicer, execute the script test_slicer.py.

.. code-block:: bash 
    
    python test_slicer.py TODO

This example will generate an LWE instance of dim XXX, BKZ-reduce it with block size XXX, run siever on the full lattice (bdgl2 algorithm), generate XXX targets with approximation factor XXX, and execute Babai's algorithm from FPyLLL and the Randomized Slicer on the instance.
It outputs the number of successful CVP runs for Babai and for the Slicer.


Running the Hybrid attack
==========================
To run the hybrid attack on LWE with parameters ``n=TODO, q=TODO``  first execute preprocessing

.. code-block:: bash 
    
    python prepocessing.py TODO

The script generates XXX different LWE ``A``'s and XXX different ``b``'s for each ``A`` with secret and error distribution 


Reproducing the experiments from the paper
====================


Reproducing Figure 1
---------------------


Reproducing Figure 2
---------------------
To get the necessary data for figure reproduction, first reproduce the figure 1. Then copy ``lwe_histo.sage`` to the root folder of the repository. Then, execute:

.. code-block:: bash 
    
    sage lwe_histo.sage

The script will tell the names XXX.png the resulting plots are stored under.

Reproducing Figure 4
---------------------
To get the necessary data for figure reproduction, run ``cvpp_exp.py`` as:

.. code-block:: bash 
    
    python cvpp_exp.py --n 70 --betamax 55 --nlats 10 --ntests 10
    python cvpp_exp.py --n 80 --betamax 55 --nlats 10 --ntests 10

This will BKZ reduce 10 lattices and launch 3*11*10*10 experiments for 3 n_randomizations 11 approximation factors, 10 lattices with 10 instances per each one. 
Once the experiments are finished, make the figures as:

.. code-block:: bash 
    
    sage cvpp_graph.sage
    
The script will tell the names XXX.png the resulting plots are stored under.

Reproducing Figure 5
---------------------



Helper scripts
====================
