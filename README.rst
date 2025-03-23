Caliban
=======

A set of C++ routines for intrinsic and extrinsic calibration.

The implemented functionality roughly corresponds to the `r1` method described in
`Solving the Robot-World Hand-Eye(s) Calibration Problem with Iterative Methods <https://arxiv.org/abs/1907.12425>`_
by Amy Tabb and Ahmad Yousef.
Additionally, it allows refining the 3D structure of the calibration object as per
`More accurate pinhole camera calibration with imperfect planar target <https://elib.dlr.de/71888/1/strobl_2011iccv.pdf>`_
by Klaus Strobl and Gerd Hirzinger (within intrinsic calibration), and optimizing the target scale (within hand-eye calibration).

Main traits and features
-------------------------

- Intrinsic calibration:
  
  - Pinhole camera with Brown-Conrady distortion model
  - Calibration object: roughly planar (:math:`Z = 0`) object - checkerboard, ChArUco board, etc.
  - Automatic choice of base points for the release object method

Intrinsic calibration roughly corresponds to the `calibrateCameraRO` function from OpenCV.
Still the `ceres-solver <http://ceres-solver.org/>`_ used for the optimization provides
more stable and faster convergence.

- Hand-eye calibration:
    
  - Eye-in-hand and eye-to-hand configurations
  - Calibration refinement based on the minimzation of reprojection error (`Shah method <https://www.researchgate.net/publication/275087810_Solving_the_Robot-WorldHand-Eye_Calibration_Problem_Using_the_Kronecker_Product>`_ used as initial guess)
  - Optimization of the target scale (as a post-step to calibration object structure refinement)

- Libraries used:
    
    - `OpenCV <https://opencv.org/>`_ for image processing, generating initial guesses for intrinsic parameters and `SE(3)` transformations
    - `Eigen <http://eigen.tuxfamily.org/>`_ for linear algebra (used in OpenCV and ceres-solver)
    - `ceres-solver <http://ceres-solver.org/>`_ for optimization

Repository structure
----------------------

Main parts of the repo are:

- **include** / **src** - the core library code, contains the implementation of intrinsic and hand-eye calibration. You can find the detailed description of the classes and functions in the corresponding header files in **include** directory.
- **app** - example application that demonstrates the usage of the library (only intrinsic calibration and checkerboard target supported)
- **tests** - unit tests for the library
- **datasets** - datasets for the example application and unit tests
- **patterns** - images and python script to generate synthetic datasets for unit tests

How to build
------------

The project uses CMake as a build system. All dependencies except for OpenCV are downloaded and built automatically
via `CPM Cmake <https://github.com/cpm-cmake/CPM.cmake>`_.

The project is built with a sequence of commands standard for CMake projects:

.. code-block:: bash

    cd <repo root>
    cmake -DCMAKE_BUILD_TYPE=Debug -DOpenCV_DIR=<path to the directory containing OpenCVConfig.cmake>  -B build/debug -S .
    cmake --build build/debug -j8

How to use
----------

Library
~~~~~~~

Right now, there is no integration with any package manager. Simply copy the **include** and **src** directories to your project.
It is responsibility of the user to take care of the dependencies.

Unit tests
~~~~~~~~~~

.. code-block:: bash

    cd <repo root>/build/debug/tests
    ctest

or 

.. code-block:: bash

    cd <repo root>/build/debug/tests
    ./intrinsic_tests
    ./extrinsic_tests

Example application
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    cd <repo root>/build/debug/app
    ./calib_app --help

This command will call built-in help for the application.
