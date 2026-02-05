
# Installation

Pre-built binaries of this API can be installed from [PyPi](https://pypi.org/project/isx) for the supported platforms.

```bash
pip install isx
```

## Installation on Apple Silicon

::: {attention}
Currently, pyisx is only supported for x86 architectures, which can be problematic, specifically on the newer Mac computers with Apple Silicon. The following section describes one option for how to use this package on newer Mac computers.
:::

For usage with Apple Silicon, the Rosetta software must be installed, and the Terminal app must be configured to use this software for automatic translation of x86 binaries to arm64. Read more [here](https://support.apple.com/en-us/102527) on how to configure Rosetta on Mac computers.

Next, install an x86 version of python with the following steps.

First, in the Rosetta terminal, install Homebrew, which is a package manager for macOS. This will install Homebrew to the `/usr/local` directory, indicating an x86 installation:
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Next, in the Rosetta terminal, install the desired python version using the x86 installation of brew:
```bash
/usr/local/bin/brew install python@3.13
```

Verify the installation worked correctly by running the following command:
```bash
python3.13 -c "import sysconfig; print(sysconfig.get_platform())"
```

The output will be `macosx-13.0-x86_64` if running the x86 version of python, otherwise the output will instead be `macosx-13.0-arm64` if running the arm version of python.

If the output indicates the x86 version of python, you can install the `pyisx` package.
For example, the following snippet demonstrates how to create a [virtual environment](https://docs.python.org/3/library/venv.html) using the x86 version of python, and then install the pyisx package within the virtual environment.
```bash
python3.13 -m venv venv
source venv/bin/activate && python -m pip install pyisx
```

## Supported Platforms

This package has been built and tested on the following operating systems, for python versions `3.9 - 3.13`:

|  OS | Version | Architecture |
|  --------- | ------- | ----- |
| macOS   | 13 | x86_64 |
| Ubuntu (Linux) | 20.04 | x86_64 |
| Windows | 11 | amd64 |

## Development (advanced)

In order to build the package locally, follow this [Development Guide](https://github.com/inscopix/pyisx?tab=readme-ov-file#development-guide).
