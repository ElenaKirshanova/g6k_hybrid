******************************
The Randomized Slicer in the General Sieve Kernel (G6K) library
******************************

The Randomized Slicer is a C++ and Python extension of the `G6K library <https://github.com/fplll/g6k>`_ that implements the batch-CVP algorithm from Doulgerakis-Laarhoven-de Weger `"Finding closest
lattice vectors using approximate Voronoi cells" <https://eprint.iacr.org/2016/888.pdf>`_.

The code is based on BDGL implementation from Ducas-Stevens-van Woerden `"Advanced lattice  sieving on GPUs, with tensor cores" <https://eprint.iacr.org/2021/141.pdf>`_.

Building the library
====================

This code has been tested on Ubuntu-\{22,24\}.04 and Windows 11 (via WSL, same distributions) which are the target platform for the code in this repository. MacOS is not supported but may work (see the instructions at the bottom of the README).

You will need the `G6K library <https://github.com/fplll/g6k>`_. For reproducing figures you would also need `SageMath <https://www.sagemath.org/>`_ and conda with conda-forge channel enabled. Enabling the channel can be done with the following command:

.. code-block:: bash

    conda config --add channels conda-forge

Steps specific to Ubuntu
---------------------

First, uptate the packages on your system ``sudo apt-get update`` and install autotools, automake, libtool and other dependencies:

.. code-block:: bash

    sudo apt-get install make pkg-config autoconf autotools-dev libtool automake gcc g++ libgmp3-dev libqd-dev libmpfr-dev


Then make sure the python has development headers installed:

.. code-block:: bash

    sudo apt install -y build-essential python3-dev

Further steps for both Ubuntu and MacOS
---------------------

If you have Ubuntu, proceed with the steps below.
If you have  g++ compiler installed from homebrew you may have issues building the code. If your only compiler is the one provided by Apple, you should be able to skip some of the steps.

1. Create conda environment

.. code-block:: bash

    conda create --name g6x
    conda activate g6x

2. Install required packages (see requirements.txt)

.. code-block:: bash

    conda install fpylll cython cysignals flake8 ipython numpy begins pytest requests scipy multiprocessing-logging matplotlib autoconf automake libtool

3. Attempt to build the code

.. code-block:: bash

    python setup.py build_ext --inplace

4. In case a compiler other than Apple’s clang is used and building fails, use Apple’s clang. Otherwise, skip the following three steps and execute tests

.. code-block:: 

    make clean
    ./configure CXX=/usr/bin/g++
    python setup.py build_ext --inplace

5. Check is building succeeded by executing tests

.. code-block:: bash

    python -m pytest


Running RandomizedSlicer
====================
To test-run our randomized slicer, execute the script test_slicer.py

.. code-block:: bash 
    
    python test_slicer.py --n 60 --betamax 55 --nexp 3 --approx_factor 0.99

This example will generate an LWE instance of dim 60, BKZ-reduce it with block size 55, run siever on the full lattice (bdgl2 algorithm), generate 3 targets with approximation factor 0.99, and execute Babai's algorithm from FPyLLL and the Randomized Slicer on the generated instances.
It outputs the number of successful CVP runs for Babai and for the Slicer.


Running the Hybrid attack
==========================

Preprocessing
--------------

To run the hybrid attack on LWE with parameters ``n=130, q=3329`` and ``kappa=4`` (the number of guessed coordinates)  first execute preprocessing

.. code-block:: bash 
    
    python preprocessing.py --params "[(130, 4, 46)]" --q 3329 --dist "ternary" --dist_param 0.08333 --recompute_instance

Here, ``--dist "ternary"`` and ``dist_param 0.08333`` corresponds to ternary secrets/errors of Hamming weight 1/6 ("ternary" and "binomial" ``dist`` supported). ``params`` is a list of triples (n, n_guess_coordinates, bkzbeta) = (LWE dimension, number of guessed coordinates, preprocessing BKZ blocksize). The preprocessing will iterate through this list.
``q`` is the LWE modulus. ``recompute_instance`` deletes all cashed data (such as reduced bases).

The script terminates within a few minutes on a laptop. It creates a report file ``lwe_instances/reduced_lattices/report_prehyb_130_3329_ternary_0.08333_0_4_46_47_46.pkl"``

The additional flag ``inst_per_lat X`` will generate ``X`` LWE ``b``'s for the same LWE matrix ``A``, the flag ``lats_per_dim Y``will generate ``Y`` difference LWE matrices ``A``. 

To parallelize BKZ reduction, add flag ``--nthreads``, to parallelize over different experiments add flag ``--nworkers``. For central binomial secrets and errors with parameter X use ``--dist "binomial" --dist_param X``. 
For example, to run 2 experiments in parallel on 3 threads each run:

.. code-block:: bash 
    
    python preprocessing.py --params "[(130, 4, 46)]" --q 3329 --dist "ternary" --dist_param 0.08333 --recompute_instance --inst_per_lat 1 --lats_per_dim 2 --nworkers 2 --nthreads 3

Note: the experiments are parallelized over ``lats_per_dim`` lattices.

Optional parameters:

* ``beta_bkz_offset`` BKZ-beta reduced bases will be computed for beta in [bkzbeta,...,bkzbeta+beta_bkz_offset] where bkzbeta is defined by the current triple from ``params`` (default ``1``)
* ``sieve_dim_max_offset`` sieving will take place in dimensions up to bkzbeta + sieve_dim_max_offset (default ``1``)
* ``nsieves`` sieving will take place in dimensions starting from bkzbeta + sieve_dim_max_offset - nsieves (default ``1``)
* ``recompute_instance`` recomputes new LWE instances (default False). Execute with this flag if LWE instance was not generated before

Progressive Hybrid
--------------

Run the hybrid attack after the preprocessing step above is finished like so

.. code-block:: bash 

    python run_prog_hyb.py --n 130 --q 3329 --dist "ternary" --dist_param 0.0833 --n_guess_coord 4

The parameter ``--n_guess_coord`` should be identical to the second parameter in ``--params`` for ``preprocessing.py``. Same applies to the ``--lats_per_dim`` parameter (default: 1).

Optional parameters:

* ``n_slicer_coord`` the minimal slicer dimension
* ``beta_pre`` BKZ blocksize the data was preprocessed with 
* ``delta_slicer_coord``  an integer defining the upper bound on the slicer dimension as n_slicer_coord+delta_slicer_coord (default ``1``)

Running the Two-Step Primal Attack
==========================
For the sake of comparison with the hybrid attack, we implemented the two-step primal attack on Kyber.

To generate an instance and run the attack on LWE with parameters ``n=130, q=3329``, ternary error and secret distribution with sparsity parameter 0.08333 and maximum BKZ blocksize parameter 60, execute

.. code-block:: bash 
    
    python lwe_g6k.py --n 130 --q 3329 --dist "ternary" --dist_param 0.0833 --blocksizes "50:60:1" --recompute_instance

The experiments will terminate in a several minutes on a laptop with the output dumped to a file ``lwe_instances/reduced_lattices/exp_130.pkl``. If the attack is successful, the user will see the solution vector.

To parallelize BKZ reduction, add flag ``--nthreads``, to parallelize over different experiments add flag ``--nworkers``. For central binomial secrets and errors with parameter X use ``--dist "binomial" --dist_param X``.

In the last line of the output you will get a string representation of a python distionary.
Each value within the dictionary contains the following values:

#. ``walltime`` -- attack's duration
#. ``dist_bnd`` -- ratio of length of the error vector to the Gaussian Heuristic.
#. ``succ`` -- whether attack succeeded.
#. ``fail_reason`` -- the reason why attack had failed or None otherwise.
#. ``g6k_len`` -- size of the siever's database.
#. ``g6k_dim`` -- dimension of the siever.
#. ``wrong_guess_time_alg3`` -- time spent on an incorrect guess in alg3
#. ``correct_guess_time_alg3`` -- time spent on a correct guess in alg3
#. ``wrong_guess_time_alg2`` -- time spent on an incorrect guess in alg2
#. ``correct_guess_time_alg2`` -- time spent on a correct guess in alg2
#. ``overhead_tsieve`` -- time spent on sieving.

Reproducing the experiments from the paper
====================


Reproducing Figure 1
---------------------
To recompute the necessary data for figure reproduction, run ``cvpp_exp.py`` as:

.. code-block:: bash 
    
    python cvpp_exp.py --n 70 --betamax 55 --nlats 10 --ntests 10 --nthreads 10 --nworkers 10
    python cvpp_exp.py --n 80 --betamax 55 --nlats 10 --ntests 10 --nthreads 10 --nworkers 10

This will BKZ reduce 10 lattices and launch 3*11*10*10 experiments for 3 values of ``n_randomizations`` ([1, 5, 10]) 11 approximation factors ([0.9, ..., 1.0]), 10 lattices with 10 instances per each lattice occupying 100 logical cores. Note: this is a heavy computation (several days on a 128 core server).
**Alternatively**, just preform the following step (uses our precomputed data).
Then copy ``gen_figures/cvpp_graph.sage`` to the root folder of the repository. Once the experiments are finished, the figures will be generated by running:

.. code-block:: bash 
    
    sage cvpp_graph.sage


The script will output the name of the .png file with a plot.

Reproducing Figure 2
---------------------
To recompute the necessary data for figure reproduction, run

.. code-block:: bash 
    
    python tailBDD.py --n 120 --beta 55 --Nlats 5 --ntests 5 --n_uniq_targets 10  --approx_factor 0.43 --nworkers 5 --nthreads 5

This will BKZ reduce 5 dimension-120 lattices and solve 5 Batch-Tail-BDD instances each consisting of 10 BDD instances (occupying 25 logical CPU cores). Note: the computations may take several hours. This will create a file named ``tail_bdd_n{n}_b{beta}.pkl`` needed for the next step. 

To get Figure 2, copy ``tailbdd.sage`` from ``gen_gigures`` to root and run:

.. code-block:: bash 
    
    sage tailbdd.sage

The script will output the name of the .png file with a plot. 

Reproducing Figure 3
---------------------
To reproduce Figure 1 perform the hybrid attack as describe above (for an appropriate distribution (binomial and/or ternary)). The experiments are done over 10 ``lats_per_dim`` with 10 ``inst_per lat`` in the following dimensions (values of ``n``):

* [140,150,160,170] for the **binomial distribution** (``--dist \"binomial\" --dist_param 3``);
* [160,170,180,190,200,210] for the **ternary distribution** (``--dist \"ternary\" --dist_param 0.1667``);
* [170,180,190,200,210] for the **sparse distribution** (``--dist \"ternary\" --dist_param 0.0833``);

Depending on the distribution considered, copy ``gen_figures/aggr_attacks_XXX.sage`` to the root directory where ``XXX`` is ``binom`` for binomial distribution, ``sparse`` for sparse distribution and ``ternary`` for the ternary distribution.

Run the corresponding script:

.. code-block:: bash 
    
    sage aggr_attacks_XXX.sage

for an appropriate value of ``XXX``.
The script will output the name of the .png file with the plot. 

Reproducing Figure 4
---------------------
To get the necessary data for figure reproduction, run the hybrid attack as explained above. Copy ``gen_figures/lwe_histo.sage`` to the root folder of the repository. Then, execute:

.. code-block:: bash 
    
    sage lwe_histo.sage

The script will output the name of the .png file with a plot. 

Comparing Slicers
---------------------
To compare our slicer against DLvW20, run ``benchmark_slicer_dist`` for ``dist \in {“binom”, “sparse”, “ternary”}`` for this and DLvW20 slicers respectively. 

Run ``aggregate_slicer_comparison.py`` in the terminal. The script will output the comparison between the runtimes of two slicers.

Algorithms
====================
#. ``hybrid_attack.py`` -- implementation of Batched-Tail-BDD;
#. ``test_slicer.py`` -- script for showcasing slicer; 
#. ``lattice_reduction.py`` -- implementation of pump'n'jump BKZ;
#. ``benchmark_slicer_our.py`` -- runs a benchmark on various lattices for our slicer;
#. ``benchmark_slicer_ww.py`` -- runs a benchmark on various lattices for WW slicer;
#. ``cvpp_exp.py`` -- investigates CVP success rate w.r.t. the approximation factor and the number of rerandomizations;
#. ``tailBDD.sage`` -- investigates Batch-Tail-BDD success rate for our slicer; 
#. ``preprocessing.py`` -- preprocessing for the hybrid attack on LWE;
#. ``run_prog_hybrid.py`` -- hybrid attack on LWE (won't launch without preprocessing stage).
#. ``lwe_g6k.py`` -- automated two-step primal attack on lwe.

Helper scripts
====================
#. ``utils.py`` -- inner subroutines used across the repository;
#. ``global_consts.py`` -- global constants used in algorithms;
#. ``sample.py`` -- various distributions and samplers;
#. ``discretegauss.py`` -- discrete Gaussian sampler


-----------------------------------------------------------------------------------------------------------------



