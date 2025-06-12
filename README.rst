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

Fetching and Installing the `[sum25] <https://github.com/Summwer/cvp-g6k-cpu-solver>` Slicer (Ubuntu)
-----------------------

Install conda as a prerequisite. For example, as follows:

.. code-block:: bash

    mkdir -p ~/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm ~/miniconda3/miniconda.sh
    source ~/miniconda3/bin/activate
    conda init --all

This implementation of slicer relies on `libalglib` which can be installed on Ubuntu as:

.. code-block:: bash

    sudo apt-get install libalglib-dev

For machines without sudo that will be:

.. code-block:: bash

    brew install wget
    wget http://archive.ubuntu.com/ubuntu/pool/universe/a/alglib/alglib_4.0.0.orig.tar.gz
    mkdir ./alglib
    tar -xvzf alglib_4.0.0.orig.tar.gz -C ./alglib
    cd ./alglib

Then the users without sudo should fix the ``#include`` blocks in ``progressive_slicer_with_d4f.cpp`` and ``randomized_iterative_slicer.cpp`` for ``libalglib``. Change the prefixes of the paths to absolute paths to ``/path/to/g6k_hybrid/alglib/alglib-cpp/``. 

Activate a conda environment.

.. code-block:: bash

    conda create -n g6x python=3.12
    conda activate g6x

Then simply run the script that will fetch the `[sum25]` repository and install it inplace.

.. code-block:: bash

    source ./smart_install.sh


Running RandomizedSlicer
====================
To test-run our randomized slicer, execute the script test_slicer.py

.. code-block:: bash 
    
    python test_slicer.py --n 60 --betamax 55 --nexp 3 --approx_factor 0.99

This example will generate an LWE instance of dim 60, BKZ-reduce it with block size 55, run siever on the full lattice (bdgl2 algorithm), generate 3 targets with approximation factor 0.99, and execute Babai's algorithm from FPyLLL and the Randomized Slicer on the generated instances.
It outputs the number of successful CVP runs for Babai and for the Slicer alongside with the solutions.


Running the Hybrid attack
==========================

Preprocessing
--------------

To run the hybrid attack on LWE with parameters ``n=130, q=3329`` and ``kappa=4`` (the number of guessed coordinates)  first execute preprocessing

.. code-block:: bash 
    
    python preprocessing.py --params "[(130, 4, 46)]" --q 3329 --dist "ternary" --dist_param 0.08333 --recompute_instance

``--dist_param 0.08333`` corresponds to ternary secrets/errors of Hamming weight 1/6. ``params`` is a list of triples (n, n_guess_coordinates, bkzbeta). The preprocessing will iterate through this list.

The script terminates within a few minutes on a laptop. It creates a report file ``lwe_instances/reduced_lattices/report_prehyb_130_3329_ternary_0.08333_0_4_46_47_46.pkl"``

The additional flag ``inst_per_lat X`` will generate ``X`` LWE ``b``'s for the same LWE matrix ``A``, the flag ``lats_per_dim Y``will generate ``Y`` difference LWE matrices ``A``. 

To parallelize BKZ reduction, add flag ``--nthreads``, to parallelize over different experiments add flag ``--nworkers``. For central binomial secrets and errors with parameter X use ``--dist "binomial" --dist_param X``.

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

The parameter ``--n_guess_coord`` should be identical to the second parameter in ``--params`` for ``preprocessing.py``.

Optional parameters:

* ``n_slicer_coord`` the minimal slicer dimension
* ``beta_pre`` BKZ blocksize the data was preprocessed with 
* ``delta_slicer_coord``  an integer defining the upper bound on the slicer dimension as n_slicer_coord+delta_slicer_coord (default ``1``)

Running the Primal attack
==========================
For the sake of comparison with the hybrid attack, we implemented the primal attack on Kyber (Kannan's embedding) in ``primal_kyber.py``

To run the attack on LWE with parameters ``n=130, q=3329``, ternary error and secret distribution with sparsity parameter 0.08333 and maximum BKZ blocksize parameter 60, execute

.. code-block:: bash 
    
    python primal_kyber.py --ns "range(130,131,1)" --q 3329 --dist "ternary" --dist_param 0.0833 --betamax 60 --recompute_instance

The experiments will terminate in several minutes on a laptop with the output dumped in a file ``lwe_instances/reduced_lattices/exp[130]_3329_ternary_0.08330.pkl``

The additional flag ``inst_per_lat X`` will generate ``X`` LWE ``b``'s for the same LWE matrix ``A``, the flag ``lats_per_dim Y``will generate ``Y`` difference LWE matrices ``A``. 

To parallelize BKZ reduction, add flag ``--nthreads``, to parallelize over different experiments add flag ``--nworkers``. For central binomial secrets and errors with parameter X use ``--dist "binomial" --dist_param X``.



Reproducing the experiments from the paper
====================


Reproducing Figure 1
---------------------
To reproduce Figure 1:
* perform the primal attack as described above,
* perform the hybrid attack as describe above (for an appropriate distribution (binomial and/or ternary).

Depending on the distribution considered, copy ``gen_figures/aggr_attacks_{XXX}.sage`` to the root directory where XXX is ``binom`` for binomial distribution, ``sparse`` for Ternary(1/6) and ``ternary`` for Ternary(1/3).

Run the corresponding script:

.. code-block:: bash 
    
    sage aggr_attacks_{XXX}.sage

The script will output the name of the .png file with the plot. 

Reproducing Figure 2
---------------------
To get the necessary data for figure reproduction, run the hybrid attack as explained above. Copy ``gen_figures/lwe_histo.sage`` to the root folder of the repository. Then, execute:

.. code-block:: bash 
    
    sage lwe_histo.sage

The script will output the name of the .png file with a plot. 

Reproducing Figure 4
---------------------
To get the necessary data for figure reproduction, run ``cvpp_exp.py`` as:

.. code-block:: bash 
    
    python cvpp_exp.py --n 70 --betamax 55 --nlats 10 --ntests 10
    python cvpp_exp.py --n 80 --betamax 55 --nlats 10 --ntests 10

This will BKZ reduce 10 lattices and launch 3*11*10*10 experiments for 3 n_randomizations ([1, 5, 10]) 11 approximation factors ([0.9, ..., 1.0]), 10 lattices with 10 instances per each lattice. 
Then copy ``gen_figures/cvpp_graph.sage`` to the root folder of the repository. Once the experiments are finished, the figures will be generated by running:

.. code-block:: bash 
    
    sage cvpp_graph.sage


The script will output the name of the .png file with a plot. 

Reproducing Figure 5
---------------------
To get the necessary data for figure reproduction, run

.. code-block:: bash 
    
    python tailBDD.py --n 120 --beta 55 --Nlats 5 --ntests 5 --n_uniq_targets 10  --approx_factor 0.43 

This will BKZ reduce 5 dimension-120 lattices and solve 5 Batch-Tail-BDD instances each consisting of 10 BDD instances.
To get Figure 5, run:

.. code-block:: bash 
    
    sage tailBDD.sage

The script will output the name of the .png file with a plot. 

Reproducing Table 3
---------------------
To compare our slicer against DLvW20 and summver, first install the [sum] slicer as per instructions in the installation section. Next, run ``benchmark_slicer_{xxx}`` for ``xxx = our, ww`` for this and DLvW20 slicers respectively. 
For [sum] run ``benchmark_slicer_pump`` in ``cvp-g6k-cpu-solver``. This will create the ``cvp_comp`` directory that will contain the experiment data.

Run ``aggregate_slicer_comparison.py`` in the terminal. The script will output the table.

Algorithms
====================
#. ``hyb_attack_on_kyber.py`` -- implementation of Batched-Tail-BDD;
#. ``test_slicer.py`` -- script for showcasing slicer; 
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


-----------------------------------------------------------------------------------------------------------------

A workaround to solve issues building on ARM-Macs (also see `Issue #128 <https://github.com/fplll/g6k/issues/128>`_)
-----------------------------------------------------------------------------------------------------------------

If you have  g++ compiler installed from homebrew you may have issues building the code. If your only compiler is the one provided by Apple, you should be able to skip some of the steps.

1. Create conda environment

.. code-block:: bash

    conda create --name g6x
    conda activate g6x

2. Install required packages (see requirements.txt)

.. code-block:: bash

    conda install fpylll cython cysignals flake8 ipython numpy begins pytest requests scipy multiprocessing-logging matplotlib autoconf automake libtool

3. Clone the g6x git repo

.. code-block:: bash

    git clone git@github.com:fplll/g6k.git

4. Checkout arm-fixes branch

.. code-block:: bash

    git checkout --track origin/arm-fixes

5. Add modifications to file g6x/siever.pyx.

Change ``def insert_best_lift(self, scoring=(lambda index, nlen, olen, aux: True), aux=None):`` (line 1664)
to  ``def insert_best_lift(self, scoring=None, aux=None):`` . And inside this function (right the Example is finished) add

.. code-block:: bash

    if scoring==None:
          scoring = lambda index, nlen, olen, aux: True

6. Attempt to build the code

.. code-block:: bash

    python setup.py build_ext --inplace

7. In case a compiler other than Apple’s clang is used and building fails, use Apple’s clang. Otherwise, skip the following three steps and execute tests

.. code-block:: bash
    make clean
    ./configure CXX=/usr/bin/g++
    python setup.py build_ext --inplace

8. Check is building succeeded by executing tests

.. code-block:: bash

    python -m pytest



