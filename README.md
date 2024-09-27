# pyisx

`pyisx` is a python package for interacting with Inscopix data. This package encapsulates the following I/O functionality:

* Reading Inscopix files (`.isxd`, `.isxc`, `.isxb`, `.gpio`, `.imu`)
* Writing Inscopix files (`.isxd`)
* Exporting Inscopix files to third-party formats (`.mp4`, `.tiff`, `.csv`, `.hdf5`)

## Install

Pre-built binaries of this API can be installed from [PyPi](https://pypi.org/project/isx/).

```bash
pip install isx
```

## Supported Platforms

This library has been built and tested on the following operating systems:

|  OS | Version | Architecture |
|  --------- | ------- | ----- |
| macOS   | 13 | x86_64 |
| Ubuntu (Linux) | 20.04 | x86_64 |
| Windows | 11 | amd64 |

Each system has been built and tested on python versions 3.9 - 3.12.

> **Note:** For Apple Silicon (arm64 architectures), the package is currently not natively supported. However, it's possible to use anaconda to configure an x86 environment and use the project.

```
CONDA_SUBDIR=osx-64 conda create -n pyisx python=3.12
conda activate pyisx
conda config --env --set subdir osx-64
```

## Development Guide

This guide documents how to build the python package wheel locally.

1. Setup `isxcore`
Follow the setup instructions for the C++ [isxcore](https://github.com/inscopix/isxcore) repo.

2. Setup python virtual environment

Create a python virtual environment, specifying the desired python version.
This guide uses anaconda for demonstration, but other tools like virtualenv or poetry can also be used.

```
conda create -n pyisx python=3.12
conda activate pyisx
```

> **Note**: On macOS systems with Apple Silicon, the conda environment is configured differently, since `isxcore` is currently only built for x86 architectures.

```
CONDA_SUBDIR=osx-64 conda create -n pyisx python=3.12
conda activate pyisx
conda config --env --set subdir osx-64
```

3. Install build & test dependencies

Inside the virtual environment install the following dependencies:

```
conda install -y build pytest
```

> **Note**: For python 3.12 the `build` package must be installed used `pip` instead.

4. Build the package

```
make build THIRD_PARTY_DIR=/path/to/third/party/dir
```

5. Run the unit tests

```
make test THIRD_PARTY_DIR=/path/to/third/party/dir TEST_DATA_DIR=/path/to/test/data/dir
```

# Support

For any questions about this package, please contact support@inscopix.bruker.com.
