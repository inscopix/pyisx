
# Overview

The [pyisx](https://github.com/inscopix/pyisx) project is a python binding for [isxcore](https://github.com/inscopix/isxcore), a C++ API for interacting with Inscopix data.

## Install

The python package for this binding, named `isx`, is available on [pypi](https://pypi.org/project/isx/) for download.
To install `isx`, run the following command in a python environment:

```python
pip install isx
```

Please refer to the :ref:`installation` guide for more details.

## File Types

This package encapsulates the following I/O functionality:

* Reading Inscopix files (`.isxd`, `.isxb`, `.gpio`, `.imu`)
* Writing Inscopix files (`.isxd`)
* Exporting Inscopix files to third-party formats (`.mp4`, `.tiff`, `.csv`)

The following table summarizes all Inscopix file types and the functionality supported by this package:

| File Type | File Format | Read | Write | Export | Description |
| --------- | ----------- | ---- | ----- | ------ | ----------- |
| `Microscope Movie` | `.isxd` | Yes | Yes | `.mp4`, `.tiff`, `.csv` | Recording acquired from a microscope |
| `Microscope Image` | `.isxd` | Yes | Yes | `.tiff` | Image acquired from a microscope |
| `Behavior Movie` | `.isxb` | Yes | No | `.mp4` | Compressed recording acquired from a behavior camera | 
| `Cell Set` | `.isxd` | Yes | Yes | `.tiff`, `.csv` | Neural cells represented as a set of temporal activity traces and spatial footprints. A `Cell Set` is generated from a `Microscope Movie` |
| `Event Set` | `.isxd` | Yes | Yes | `.csv` | Neural events (e.g., calcium events) represented as a set of discrete signal traces. An `Event Set` is generated from a `Cell Set` |
| `Vessel Set` | `.isxd` | Yes | Yes | `.tiff`, `.csv` | Vessels represented as a set of vessel diameter or red blood cell (rbc) velocity traces. A `Vessel Set` is generated from a blood flow `Microscope Movie` |
| `GPIO`  | `.gpio`, `.isxd` | Yes | No | `.isxd`, `.csv` | General purpose input/output signals recorded from an acquisition device |
| `IMU` | `.imu` | Yes | No | `.isxd`, `.csv` | Inertial measurement unit (accelerometer, magnetometer, orientation​) recorded from an acquisition device |
| `Compressed Microscope Movie` | `.isxc` | No | No | N/A | Compressed recording acquired from a microscope. A `Compressed Microscope Movie` is decompressed into a `Microscope Movie` |

## Next Steps

To learn more about how to use the `isx` package, refer to the :ref:`examples` guide and the :ref:`reference`.