# pyisx

`isx` is a python package for interacting with Inscopix data.
This package encapsulates the following I/O functionality:
* Reading Inscopix files (`.isxd`, `.isxb`, `.gpio`, `.imu`)
* Writing Inscopix files (`.isxd`)
* Exporting Inscopix files to third-party formats (`.mp4`, `.tiff`, `.csv`)

The `isx` package is built from the `pyisx` project, a python binding for [isxcore](https://github.com/inscopix/isxcore), a C++ API for interacting with Inscopix data.

## Documentation

For help, please refer to the [documentation](https://github.com/inscopix/pyisx).

## Install

Pre-built binaries of this API can be installed from [PyPi](https://pypi.org/project/isx/).

```bash
pip install isx
```

> **Note:** For Apple Silicon (i.e., macOS arm64 architecture), the package is currently not natively supported. However, it's possible to use anaconda to configure an x86 environment and use the project.

```bash
CONDA_SUBDIR=osx-64 conda create -n <name> python=<python>
conda activate <name>
conda config --env --set subdir osx-64
pip install isx
```

Replace `<name>` with a name for the conda environment, and `<python>` with the python version to use.

## Supported Platforms

This library has been built and tested on the following operating systems, for python versions `3.9 - 3.12`.

|  OS | Version | Architecture |
|  --------- | ------- | ----- |
| macOS   | 13 | x86_64 |
| Ubuntu (Linux) | 20.04 | x86_64 |
| Windows | 11 | amd64 |


## Development Guide

This guide documents how to build the python package wheel locally.

1. Clone the repo

Setup the repo and initialize its submodule:

```bash
git clone git@github.com:inscopix/pyisx.git
git submodule update --init
```

1. Setup `isxcore`
Follow the setup instructions for the C++ [isxcore](https://github.com/inscopix/isxcore) repo.

2. Setup python virtual environment

Create a python virtual environment, specifying the desired python version.
This guide uses anaconda for demonstration, but other tools like virtualenv or poetry can also be used.

```bash
conda create -n <name> python=<python>
conda activate <name>
```

Replace `<name>` with a name for the conda environment, and `<python>` with the python version to use.

> **Note**: On macOS systems with Apple Silicon, the conda environment is configured differently, since `isxcore` is currently only built for x86 architectures.

```bash
CONDA_SUBDIR=osx-64 conda create -n <name> python=<python>
conda activate <name>
conda config --env --set subdir osx-64
```

Replace `<name>` with a name for the conda environment, and `<python>` with the python version to use.

3. Install build & test dependencies

Inside the virtual environment install the following dependencies:

```bash
conda install -y build pytest
```

> **Note**: For python 3.12 the `build` package must be installed used `pip` instead.

4. Build the package

```bash
make build THIRD_PARTY_DIR=/path/to/third/party/dir
```

5. Run the unit tests

```bash
make test THIRD_PARTY_DIR=/path/to/third/party/dir TEST_DATA_DIR=/path/to/test/data/dir
```

# Support

For any questions or bug reports, please open an issue in our [issue tracker](https://github.com/inscopix/pyisx/issues).
