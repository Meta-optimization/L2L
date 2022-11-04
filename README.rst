L2L Gradient-free Optimization Framework
++++++++++++++++++++++++++++++++++++++++

.. image:: https://travis-ci.org/Meta-optimization/L2L.svg?branch=master
    :target: https://travis-ci.org/Meta-optimization/L2L

.. image:: https://coveralls.io/repos/github/Meta-optimization/L2L/badge.svg?branch=master
    :target: https://coveralls.io/github/Meta-optimization/L2L?branch=master


About
*****

The L2L (Learning-to-learn) gradient-free optimization framework contains well documented and tested implementations of various gradient free optimization algorithms. It also defines an API that makes it easy to optimize (hyper-)parameters for any task (optimizee). All the implementations in this package are parallel and can run across different cores and nodes (but equally well on a single core).

NOTE: The L2L framework is currently in **BETA**

Information
***********
This branch is accompanying the submission `Exploring hyper-parameter spaces of neuroscience models on high performance computers with Learning to Learn <https://arxiv.org/abs/2202.13822>`_. It will stay in a frozen shape to represent the use cases conducted in the paper.

Additionally to the requirements to install and run L2L, some libraries, especially simulators, have to be set up to successfully execute the use cases presented in the paper:

- `NEST>=3.0 <https://nest-simulator.readthedocs.io/en/v3.0/>`_
- `NetLogo v6.5 <https://ccl.northwestern.edu/netlogo/>`_
- `Arbor <https://arbor-sim.org/>`_
- `TVB <https://docs.thevirtualbrain.org/>`_
- OpenAiGym==0.18.0
- PyTorch>=1.7.0
- SDICT

The scripts to run the examples in the manuscript are located in the `bin <https://github.com/Meta-optimization/L2L/tree/frontiers_submission/bin>`_ folder. Please read the instructions in the `README <https://github.com/Meta-optimization/L2L/blob/frontiers_submission/bin/README.md>`_ file as well.

Getting Started
***************

If you are developing a new Optimizee or want to try out a new Optimizee with the Optimizers in the L2L package, install
L2L as a python package. See section `Installing the L2L Package`_ for details on how to install the package (this
automatically installs all requirements).

Documentation is available at `<https://meta-optimization.github.io/L2L/>`_.


Installing the L2L Package
**************************

From the Top-Level directory of the directory, run the following command:

    python -m pip install -e . 
    
or alternatively (if you do not use a virtual environment) 
    
    python -m pip install -e . --user

*The `--user` flag is to be used if you wish to install in the user path as opposed
to the root path*. However, we **recommend to use a virtual enviornment**, 
such as the standard Python env or conda env. 

The above will install the package by creating symlinks to the code files in the
relevant directory containing python modules. This means that you can change any
of the code files and see the changes reflected in the package immediately (i.e.
without requiring a reinstall). In order to uninstall one may run the following:

    pip uninstall L2L 

*Note that if the setup was done using sudo access, then the uninstall must also
be done using sudo access*

Having installed this package, we now have access to the top level `l2l` module
which contains all the relevant modules relevant for using the l2l package.

This should also install the `sphinx` package which should now enable you to build
the documentation as specified below.


Building Documentation
**********************
Run the following command from the `doc` directory

    make html

And open the documentation with

   firefox _build/html/index.html

All further (and extensive) documentation is in the html documentation!


Invoking the tests
******************

To run the tests go to the folder `l2l/tests` and execute: 

    python test_all.py
    
If the package nosetests is installed run on the top folder (L2L):

     nosetests -v --with-coverage --cover-package=l2l/
