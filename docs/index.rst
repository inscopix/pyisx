
pyisx documentation
====================

The `pyisx <https://github.com/inscopix/pyisx>`_ project is a python binding for `isxcore <https://github.com/inscopix/isxcore>`_, a C++ API for interacting with Inscopix data.
The python package for this binding, named :code:`isx`, is available on `pypi <https://pypi.org/project/isx/>`_ for download.
This package encapsulates the following I/O functionality:

* Reading Inscopix files (:code:`.isxd`, :code:`.isxb`, :code:`.gpio`, :code:`.imu`)
* Writing Inscopix files (:code:`.isxd`)
* Exporting Inscopix files to third-party formats (:code:`.mp4`, :code:`.tiff`, :code:`.csv`)


.. toctree::
   :maxdepth: 1
   :caption: Contents:

   overview.rst
   installation.rst
   reference.rst
   examples.rst
   contributing.rst

Quick Start
-----------

To install :code:`isx`, run the following command in a python environment:

.. code-block:: python

   pip install isx

License 
-------

This project has been released under a `CC BY-NC license <https://creativecommons.org/licenses/by-nc/4.0>`_. This means that you are free to:

* Share — copy and redistribute the material in any medium or format
* Adapt — remix, transform, and build upon the material

   * The licensor cannot revoke these freedoms as long as you follow the license terms.

Under the following terms:

* Attribution — You must give appropriate credit , provide a link to the license, and indicate if changes were made . You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
* NonCommercial — You may not use the material for commercial purposes .
* No additional restrictions — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.
