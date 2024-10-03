.. _overview:

Overview
**********

The `pyisx <https://github.com/inscopix/pyisx>`_ project is a python binding for `isxcore <https://github.com/inscopix/isxcore>`_, a C++ API for interacting with Inscopix data.

Install
-------

The python package for this binding, named :code:`isx`, is available on `pypi <https://pypi.org/project/isx/>`_ for download.
To install :code:`isx`, run the following command in a python environment:

.. code-block:: python

   pip install isx

Please refer to the :ref:`installation` guide for more details.

.. _fileTypes:

File Types
----------

This package encapsulates the following I/O functionality:

* Reading Inscopix files (:code:`.isxd`, :code:`.isxb`, :code:`.gpio`, :code:`.imu`)
* Writing Inscopix files (:code:`.isxd`)
* Exporting Inscopix files to third-party formats (:code:`.mp4`, :code:`.tiff`, :code:`.csv`)

The following table summarizes all Inscopix file types and the functionality supported by this package:

.. list-table::
    
    * - File Type
      - File Format
      - Read
      - Write
      - Export
      - Description
    * - `Microscope Movie`
      - :code:`.isxd`
      - Yes
      - Yes
      - :code:`.mp4`, :code:`.tiff`, :code:`.csv`
      - Recording acquired from a microscope
    * - `Microscope Image`
      - :code:`.isxd`
      - Yes
      - Yes
      - :code:`.tiff`
      - Image acquired from a microscope
    * - `Behavior Movie`
      - :code:`.isxb`
      - Yes
      - No
      - :code:`.mp4`
      - Compressed recording acquired from a behavior camera
    * - `Cell Set`
      - :code:`.isxd`
      - Yes
      - Yes
      - :code:`.tiff`, :code:`.csv`
      - Neural cells represented as a set of temporal activity traces and spatial footprints. A `Cell Set` is generated from a `Microscope Movie`
    * - `Event Set`
      - :code:`.isxd`
      - Yes
      - Yes
      - :code:`.csv`
      - Neural events (e.g., calcium events) represented as a set of discrete signal traces. An `Event Set` is generated from a `Cell Set`
    * - `Vessel Set`
      - :code:`.isxd`
      - Yes
      - Yes
      - :code:`.tiff`, :code:`.csv`
      - Vessels represented as a set of vessel diameter or red blood cell (rbc) velocity traces. A `Vessel Set` is generated from a blood flow `Microscope Movie`
    * - `GPIO`
      - :code:`.gpio`, :code:`.isxd`
      - Yes
      - No
      - :code:`.isxd`, :code:`.csv`
      - General purpose input/output signals recorded from an acquisition device
    * - `IMU`
      - :code:`.imu`
      - Yes
      - No
      - :code:`.isxd`, :code:`.csv`
      - Inertial measurement unit (accelerometer, magnetometer, orientation​) recorded from an acquisition device
    * - `Compressed Microscope Movie`
      - :code:`.isxc`
      - No
      - No
      - N/A
      - Compressed recording acquired from a microscope. A `Compressed Microscope Movie` is decompressed into a `Microscope Movie`

Next Steps
----------

To learn more about how to use the :code:`isx` package, refer to the :ref:`examples` guide and the :ref:`reference`.
