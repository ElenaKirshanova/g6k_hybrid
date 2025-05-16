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
    
    python test_slicer.py -n 60 --betamax 55 --nexp 3 --approx_factor 0.99

This example will generate an LWE instance of dim 60, BKZ-reduce it with block size 55, run siever on the full lattice (bdgl2 algorithm), generate 3 targets with approximation factor 0.99, and execute Babai's algorithm from FPyLLL and the Randomized Slicer on the instance.
It outputs the number of successful CVP runs for Babai and for the Slicer.


Running the Hybrid attack
==========================

Preprocessing
--------------

To run the hybrid attack on LWE with parameters ``n=130, q=3329`` and ``kappa=4`` (the number of guessed coordinates)  first execute preprocessing

.. code-block:: bash 
    
    python preprocessing.py --params "[(130, 4, 46)]" --q 3329 --dist "ternary" --dist_param 0.08333

``--dist_param 0.08333`` corresponds to ternary secrets/errors of Hamming weight 1/6.

The script terminates within a few minutes on a laptop. It creates a report file ``lwe_instances/reduced_lattices/report_prehyb_130_3329_ternary_0.8333_0_4_46_47_46.pkl"``

Optional parameters:

* ``beta_bkz_offset`` TODO (default ``1``)
* ``sieve_dim_max_offset`` TODO (default ``1``)
* ``recompute_instance`` TODO (default False)
* TODO: add the new parameter

Progressive Hybrid
--------------

Optional parameters:

* ``n_slicer_coord`` TODO (default ``1``)
* ``delta_slicer_coord`` TODO (default ``1``)
* TODO: add the new parameter

Running the Primal attack
==========================
For the sake of comparison with the hybrid attack, we implemented the primal attack on Kyber (Kannan's embedding) in ``primal_kyber.py``

To run the attack on LWE with parameters ``n=130, q=3329``, ternary error and secret distribution with sparsity parameter 0.8333 and maximum BKZ blocksize parameter 60, execute

.. code-block:: bash 
    
    python primal_kyber.py --ns "range(130,131,1)" --q 3329 --dist "ternary" --dist_param 0.833 --betamax 60

The experiments will terminate in an hour on a laptop with the output dumped in a file ``lwe_instances/reduced_lattices/exp{[n]}_{q}_{dist}_{dist_param}.pkl``

The additional flag ``inst_per_lat X`` will generate ``X`` LWE ``b``'s for the same LWE matrix ``A``, the flag ``lats_per_dim Y``will generate ``Y`` difference LWE matrices ``A``. 

To parallelize BKZ reduction, add flag ``--nthreads``, to parallelize over different experiments add flag ``--nworkers``. For central binomial secrets and errors with parameter X use ``--dist "binomial" --dist_param X``.





Reproducing the experiments from the paper
====================


Reproducing Figure 1
---------------------


Reproducing Figure 2
---------------------
To get the necessary data for figure reproduction, first reproduce the figure 1. Then copy ``gen_figures/lwe_histo.sage`` to the root folder of the repository. Then, execute:

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
Then copy ``gen_figures/cvpp_graph.sage`` to the root folder of the repository. Once the experiments are finished, make the figures as:

.. code-block:: bash 
    
    sage cvpp_graph.sage


The script will tell the names XXX.png the resulting plots are stored under.

Reproducing Figure 5
---------------------
To get the necessary data for figure reproduction, run
.. code-block:: bash 
    
    python tailBDD.py --n 120 --beta 55 --Nlats 5 --ntests 5 --n_uniq_targets 10  --approx_factor 0.43 

This will BKZ reduce 5 dimension-120 lattices and solve 5 Batch-Tail-BDD instances each consisting of 10 BDD instances.
To get the figure 5, run:

.. code-block:: bash 
    
    sage tailBDD.sage

Algorithms
====================
#. ``hyb_attack_on_kyber.py`` -- implementation of Batched-Tail-BDD;
#. ``test_slicer `` -- script for showcasing slicer; 
#. ``lattice_reduction.py`` -- implementation of pump'n'jump BKZ;
#. ``benchmark_slicer_our.py`` -- runs a benchmark on various lattices for our slicer;
#. ``cvpp_exp.py`` -- investigates CVP success rate w.r.t. the approximation factor and the number of rerandomizations;
#. ``tailBDD.sage`` -- investigates Batch-Tail-BDD success rate for our slicer; 
#. ``primal_kyber.py`` -- primal attack on LWE;
#. ``preprocessing.py`` -- preprocessing for the hybrid attack on LWE;
#. ``run_prog_hybrid.py`` -- hybrid attack on LWE (won't launch without preprocessing stage).

Helper scripts
====================
#. ``utils.py`` -- inner subroutines used across the repository;
#. ``global_consts.py`` -- global constants used in algorithms;
#. ``sample.py`` -- various distributions and samplers;
#. ``discrete_gaussian.py`` -- discrete Gaussian sampler
