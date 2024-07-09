# isx: A Python API for interacting with Inscopix data

This is a python API for reading and writing Inscopix files.

## Install

Pre-built binaries of this API can be installed from PyPi

```bash
pip install isx
```

## Supported Platforms

This library has been built and tested on the following operating systems:

|  OS | Version |
|  --------- | ------- |
| macOS   | 13 |
| Ubuntu (Linux) | 20.04 |
| Windows | 11 |

Each system has been built and tested with the following versions of python:
- 3.9
- 3.10
- 3.11
- 3.12

## Development Guide

1. Setup `isxcore`
Follow the setup instructions for the C++ `isxcore` repo: https://github.com/inscopix/isxcore

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
