.. _installation:

Installation
*************

Pre-built binaries of this API can be installed from `PyPi <https://pypi.org/project/isx>`_ for the supported platforms.

.. code-block:: bash
   
   pip install isx

.. warning::
   
   For Apple Silicon (i.e., macOS with arm64 architecture), the package is currently not natively supported. However, it's possible to use `anaconda <https://www.anaconda.com/>`_ to configure an x86 environment and use the project.

.. code-block:: bash

   CONDA_SUBDIR=osx-64 conda create -n <name> python=3.12
   conda activate <name>
   conda config --env --set subdir osx-64
   pip install isx

Replace `<name>` with a name for the conda environment.

Supported Platforms
-------------------

This package has been built and tested on the following operating systems, for python versions :code:`3.9 - 3.12`:

.. list-table::

   * - OS
     - Version
     - Architecture
   * - macOS
     - 13
     - x86_64
   * - Linux
     - Ubuntu 20.04
     - x86_64
   * - Windows
     - 11
     - amd64
