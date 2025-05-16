******************************
The Randomized Slicer in the General Sieve Kernel (G6K) library
******************************

The Randomized Slicer is a C++ and Python extension of the `G6K library<https://github.com/fplll/g6k>` that implements the batch-CVP algorithm from




Building the library
====================

You will need the current master of FPyLLL. See ``bootstrap.sh`` for creating (almost) all dependencies from scratch:

.. code-block:: bash

    # once only: creates local python env, builds fplll, fpylll and G6K
    ./bootstrap.sh [ -j # ]
    
    # for every new shell: activates local python env
    source ./activate                   

On systems with co-existing python2 and 3, you can force a specific version installation using ``PYTHON=<pythoncmd> ./boostrap.sh`` instead.
The number of parallel compilation jobs can be controlled with `-j #`.

If building via ```./bootstrap.sh``` fails, then the script will return an error code. 
The error codes are documented in ```bootstrap.sh.```

Otherwise, you will need fplll and fpylll already installed and build the G6K Cython extension like so:

.. code-block:: bash

    pip install Cython
    pip install -r requirements.txt
    python setup.py build_ext --inplace [ -j # ]

This builds G6K **in place**. Alternatively, you can skip ```--inplace``` and run ```python setup.py install``` as usual after building.
    
It's possible to alter the C++ kernel build configuration as follows:

.. code-block:: bash

    make clean
    ./configure [opts...]           # e.g. opts: --enable-native --enable-templated-dim --with-max-sieving-dim=128
                                    # see ./configure --help for more options
    python setup.py build_ext [ -j # ]


Running RandomizedSlicer
====================
In order to test-run our randomized slicer, run the script test_slicer.py.


Reproducing the experiments from the paper
====================



# Reproducing Figure 2
...

