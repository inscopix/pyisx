
# Installation

Pre-built binaries of this API can be installed from [PyPi](https://pypi.org/project/isx) for the supported platforms.

```bash
pip install isx
```

::: {attention}
For Apple Silicon (i.e., macOS with arm64 architecture), the package is currently not natively supported. However, it's possible to use [anaconda](https://www.anaconda.com/) to configure an x86 environment and use the project.
:::

```bash
CONDA_SUBDIR=osx-64 conda create -n <name> python=<python>
conda activate <name>
conda config --env --set subdir osx-64
pip install isx
```

Replace `<name>` with a name for the conda environment, and `<python>` with the python version to use.

## Supported Platforms

This package has been built and tested on the following operating systems, for python versions `3.9 - 3.12`:

|  OS | Version | Architecture |
|  --------- | ------- | ----- |
| macOS   | 13 | x86_64 |
| Ubuntu (Linux) | 20.04 | x86_64 |
| Windows | 11 | amd64 |

## Development (advanced)

In order to build the package locally, follow this [Development Guide](https://github.com/inscopix/pyisx?tab=readme-ov-file#development-guide).
