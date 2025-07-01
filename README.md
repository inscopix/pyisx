# pyisx

`isx` is a python package for interacting with Inscopix data.
This package encapsulates the following I/O functionality:
* Reading Inscopix files (`.isxd`, `.isxb`, `.gpio`, `.imu`)
* Writing Inscopix files (`.isxd`)
* Exporting Inscopix files to third-party formats (`.mp4`, `.tiff`, `.csv`)

The `isx` package is built from the `pyisx` project, a python binding for [isxcore](https://github.com/inscopix/isxcore), a C++ API for interacting with Inscopix data.

## Documentation

For help, please refer to the [documentation](https://inscopix.github.io/pyisx/).

## Install

Pre-built binaries of this API can be installed from [PyPi](https://pypi.org/project/isx/).

```bash
pip install isx
```

> **Note**: Currently, pyisx is only supported for x86 architectures, which can be problematic, specifically on the newer Mac computers with Apple Silicon. For usage with Apple Silicon, the Rosetta software must be installed, and the Terminal app must be configured to use this software for automatic translation of x86 binaries to arm64. Read more [here](https://support.apple.com/en-us/102527) on how to configure Rosetta on Mac computers.

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

Create a python virtual environment using venv, specifying the desired python version.

```bash
make env PYTHON=python3.13
```

> **Note**: Currently, pyisx is only supported for x86 architectures, which can be problematic, specifically on the newer Mac computers with Apple Silicon. For usage with Apple Silicon, the Rosetta software must be installed, and the Terminal app must be configured to use this software for automatic translation of x86 binaries to arm64. Read more [here](https://support.apple.com/en-us/102527) on how to configure Rosetta on Mac computers.

3. Build the package

Once the virtual environment is setup, the package can be built.

```bash
make build THIRD_PARTY_DIR=/path/to/third/party/dir
```

4. Run the unit tests

```bash
make test THIRD_PARTY_DIR=/path/to/third/party/dir TEST_DATA_DIR=/path/to/test/data/dir
```

# Support

For any questions or bug reports, please open an issue in our [issue tracker](https://github.com/inscopix/pyisx/issues).
