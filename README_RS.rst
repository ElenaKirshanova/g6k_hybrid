******************************
The Randomized Slicer in the General Sieve Kernel (G6K) library
******************************

The Randomized Slicer is a C++ and Python extension of the `G6K library <https://github.com/fplll/g6k>`_ that implements the batch-CVP algorithm from Doulgerakis-Laarhoven-de Weger `"Finding closest
lattice vectors using approximate Voronoi cells" <https://eprint.iacr.org/2016/888.pdf>`_.

The code is based on BDGL implementation from Ducas-Stevens-van Woerden `"Advanced lattice  sieving on GPUs, with tensor cores" <https://eprint.iacr.org/2021/141.pdf>`_

Building the library
====================

You will the `G6K library <https://github.com/fplll/g6k>`_ . Building on Lunix usually works by running ``bootstrap.sh`` (see comprehensive instruction at the `G6K repository <https://github.com/fplll/g6k>`_):

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

The script generates XXX different LWE matrices and XXX different ``b``'s for each LWE matrix with secret and error distribution 

Running the Primal attack
==========================
For the sake of comparison with the hybrid attack, we implemented the primal attack on Kyber (Kannan's embedding) in ``primal_kyber.py``

To run the attack on LWE with parameters ``n=130, q=3329``, ternary error and secret distribution with spacity parameter 0.8333 and maximum BKZ blocksize parameter 60, execute

.. code-block:: bash 
    
    python primal_kyber.py --ns "range(130,131,1)" --q 3329 --dist "ternary" --dist_param 0.833 --betamax 60

The experiments will terminate in several minutes with the output:

The additional flag ``inst_per_lat X`` will generate ``X`` LWE ``b``'s for the same LWE matrix ``A``, the flag ``lats_per_dim Y``will generate ``Y`` difference LWE matrices ``A``. 

To parallellize BKZ reduction, add flag ``--nthreads``, to parallelize over different experiments add flag ``--nworkers``.





Reproducing the experiments from the paper
====================


Reproducing Figure 1
---------------------


Reproducing Figure 2
---------------------

Reproducing Figure 4
---------------------

Reproducing Figure 5
---------------------



Helper scripts
====================
